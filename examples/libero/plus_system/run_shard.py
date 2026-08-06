#!/usr/bin/env python3
"""Run one worker of one LIBERO-plus eval shard against a running policy server.

Consumes a shard file written by generate_manifest.py, evaluates the variants this
worker owns, and appends one JSON line per episode to the shard's result directory.
Re-running is idempotent: episodes already recorded by any worker are skipped.

The libero/numpy imports live inside the functions that need them so the bookkeeping
logic here stays importable (and testable) without a simulator install.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LIBERO_EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[1]  # holds episode_runner, imported flat
FINGERPRINT_FILE = "fingerprint.txt"
EPISODE_FILE_GLOB = "episodes_worker*.jsonl"
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def load_shard_spec(shard_file: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(shard_file).read_text())


def resolve_result_dir(spec: dict, repo_root: pathlib.Path) -> pathlib.Path:
    """Shard specs store result_dir relative to the repo root."""
    return pathlib.Path(repo_root) / spec["result_dir"]


def check_fingerprint(result_dir: pathlib.Path, fingerprint: str) -> None:
    """Refuse a result dir that was filled by a different manifest, else claim it.

    Regenerating the manifest reshuffles variants across shards, so mixing episodes
    from two fingerprints in one directory would silently corrupt the aggregate.
    """
    path = result_dir / FINGERPRINT_FILE
    if path.is_file():
        existing = path.read_text().strip()
        if existing != fingerprint:
            raise SystemExit(
                f"Fingerprint mismatch in {result_dir}:\n"
                f"  existing results: {existing}\n"
                f"  this shard file:  {fingerprint}\n"
                "The manifest changed since those episodes were collected. Point at a fresh result dir, "
                "or regenerate with generate_manifest.py --force --discard-results."
            )
        return
    # Racing workers write identical content, so a plain write+rename is enough.
    tmp_path = result_dir / f"{FINGERPRINT_FILE}.tmp{os.getpid()}"
    tmp_path.write_text(fingerprint + "\n")
    os.replace(str(tmp_path), str(path))


def load_done_episodes(result_dir: pathlib.Path) -> set:
    """Collect (variant_name, trial_idx) from every worker's file: worker counts may change between runs."""
    done = set()
    for path in sorted(pathlib.Path(result_dir).glob(EPISODE_FILE_GLOB)):
        with path.open() as episode_file:
            for line_number, line in enumerate(episode_file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A killed job can leave a half-written final line; drop just that line.
                    logging.warning(f"Ignoring unparsable line {line_number} of {path}")
                    continue
                done.add((record["variant_name"], record["trial_idx"]))
    return done


def owned_variants(variants: list, worker_id: int, num_workers: int) -> list:
    """Stride assignment over spec order: workers together cover the shard exactly once."""
    return variants[worker_id::num_workers]


def init_state_index(trial_idx: int, num_init_states: int) -> int:
    """Layout variants ship a single init state, so extra trials wrap instead of indexing past the end."""
    return trial_idx % num_init_states


def should_record_images(save_videos: str, trial_idx: int) -> bool:
    """Whether the rollout must keep frames; skipping saves ~300 x 224x224x3 bytes per episode."""
    if save_videos == "all":
        return True
    if save_videos == "failures_first_trial":
        return trial_idx == 0  # success is only known afterwards, so record every first trial
    if save_videos == "none":
        return False
    raise ValueError(f"Unknown save_videos policy: {save_videos}")


def should_save_video(save_videos: str, trial_idx: int, success: bool) -> bool:
    if save_videos == "all":
        return True
    if save_videos == "failures_first_trial":
        return trial_idx == 0 and not success
    if save_videos == "none":
        return False
    raise ValueError(f"Unknown save_videos policy: {save_videos}")


def episode_record(
    spec: dict,
    worker_id: int,
    variant: dict,
    trial_idx: int,
    init_state_idx: int,
    success: bool,
    num_policy_steps: int,
    wall_time_s: float,
) -> dict:
    return {
        "eval_name": spec["eval_name"],
        "checkpoint_config": spec["checkpoint_config"],
        "checkpoint_step": spec["checkpoint_step"],
        "task_suite": spec["task_suite"],
        "shard_index": spec["shard_index"],
        "worker_id": worker_id,
        "variant_name": variant["name"],
        "base_task": variant["base_task"],
        "category": variant["category"],
        "difficulty": variant["difficulty"],
        "trial_idx": trial_idx,
        "init_state_idx": init_state_idx,
        "success": success,
        "num_policy_steps": num_policy_steps,
        "wall_time_s": round(wall_time_s, 2),
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": spec["config_fingerprint"],
    }


def append_record(episodes_path: pathlib.Path, record: dict) -> None:
    with episodes_path.open("a") as episode_file:
        episode_file.write(json.dumps(record) + "\n")
        episode_file.flush()


def run_worker(
    spec: dict,
    result_dir: pathlib.Path,
    variants: list,
    done_episodes: set,
    worker_id: int,
    host: str,
    port: int,
) -> None:
    """Evaluate this worker's variants; heavy imports happen here, not at module import."""
    sys.path.insert(0, str(LIBERO_EXAMPLE_DIR))
    import imageio
    from libero.libero import benchmark
    import numpy as np
    from openpi_client import websocket_client_policy

    from episode_runner import make_libero_env
    from episode_runner import run_episode

    np.random.seed(spec["seed"])
    trials_per_variant = spec["trials_per_variant"]
    save_videos = spec["save_videos"]
    episodes_path = result_dir / f"episodes_worker{worker_id:02d}.jsonl"
    video_dir = result_dir / "videos"

    owned_episodes = len(variants) * trials_per_variant
    pending = [
        (variant, [t for t in range(trials_per_variant) if (variant["name"], t) not in done_episodes])
        for variant in variants
    ]
    pending = [(variant, trials) for variant, trials in pending if trials]
    completed = owned_episodes - sum(len(trials) for _, trials in pending)
    episodes_run, successes_run = 0, 0
    logging.info(f"Worker {worker_id}: {len(variants)} variants, {completed}/{owned_episodes} episodes already done")
    if not pending:
        logging.info(f"Worker {worker_id}: nothing to do")
        return

    task_suite = benchmark.get_benchmark_dict()[spec["task_suite"]]()
    name_to_task_id = {task_suite.get_task(i).name: i for i in range(task_suite.n_tasks)}

    client = websocket_client_policy.WebsocketClientPolicy(host, port)

    for variant, pending_trials in pending:
        task_id = name_to_task_id.get(variant["name"])
        if task_id is None:
            raise SystemExit(f"Variant '{variant['name']}' is not in task suite '{spec['task_suite']}'")
        init_states = task_suite.get_task_init_states(task_id)
        env, task_description = make_libero_env(task_suite.get_task(task_id), LIBERO_ENV_RESOLUTION, spec["seed"])

        try:
            for trial_idx in pending_trials:
                init_state_idx = init_state_index(trial_idx, len(init_states))
                started_at = time.time()
                result = run_episode(
                    env,
                    client,
                    task_description,
                    init_states[init_state_idx],
                    max_steps=spec["max_steps"],
                    replan_steps=spec["replan_steps"],
                    record_images=should_record_images(save_videos, trial_idx),
                )
                wall_time_s = time.time() - started_at

                if should_save_video(save_videos, trial_idx, result.success):
                    video_dir.mkdir(parents=True, exist_ok=True)
                    suffix = "success" if result.success else "failure"
                    imageio.mimwrite(
                        video_dir / f"{variant['name']}_t{trial_idx}_{suffix}.mp4",
                        [np.asarray(image) for image in result.replay_images],
                        fps=10,
                    )

                append_record(
                    episodes_path,
                    episode_record(
                        spec,
                        worker_id,
                        variant,
                        trial_idx,
                        init_state_idx,
                        result.success,
                        result.num_policy_steps,
                        wall_time_s,
                    ),
                )

                completed += 1
                episodes_run += 1
                successes_run += int(result.success)
                logging.info(
                    "[{}/{}] {} t{} {} ({} steps, {:.1f}s) | this run: {}/{} ({:.1f}%)".format(
                        completed,
                        owned_episodes,
                        variant["name"],
                        trial_idx,
                        "SUCCESS" if result.success else "failure",
                        result.num_policy_steps,
                        wall_time_s,
                        successes_run,
                        episodes_run,
                        100.0 * successes_run / episodes_run,
                    )
                )
        finally:
            env.close()

    logging.info(f"Worker {worker_id} done: {completed}/{owned_episodes} owned episodes recorded ({episodes_run} here)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-file", type=pathlib.Path, required=True, help="Shard JSON from generate_manifest.py.")
    parser.add_argument("--worker-id", type=int, required=True, help="Index of this worker, 0-based.")
    parser.add_argument("--num-workers", type=int, required=True, help="Workers launched for this shard.")
    parser.add_argument("--host", type=str, default="localhost", help="Policy server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port.")
    parser.add_argument(
        "--repo-root", type=pathlib.Path, default=REPO_ROOT, help="Root that the spec's result_dir is relative to."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.num_workers < 1 or not 0 <= args.worker_id < args.num_workers:
        raise SystemExit(f"Invalid worker {args.worker_id} of {args.num_workers}")

    spec = load_shard_spec(args.shard_file)
    result_dir = resolve_result_dir(spec, args.repo_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    check_fingerprint(result_dir, spec["config_fingerprint"])

    variants = owned_variants(spec["variants"], args.worker_id, args.num_workers)
    done_episodes = load_done_episodes(result_dir)
    logging.info(
        "Shard {} of eval '{}' ({}@{}) -> {}".format(
            spec["shard_index"], spec["eval_name"], spec["checkpoint_config"], spec["checkpoint_step"], result_dir
        )
    )
    run_worker(spec, result_dir, variants, done_episodes, args.worker_id, args.host, args.port)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one worker of one LIBERO-plus eval shard against a running policy server.

Consumes a shard file written by generate_manifest.py, evaluates the variants this
worker owns, and appends one JSON line per episode to the shard's result directory.
Re-running is idempotent: episodes already recorded by any worker are skipped.

Episodes that end in an exception (dead server, sim crash) are NOT recorded, so a
retry picks them up; a run that hits MAX_CONSECUTIVE_ERRORS of them aborts loudly
instead of writing an infrastructure outage down as thousands of task failures.

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
import tempfile
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LIBERO_EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[1]  # holds episode_runner, imported flat
FINGERPRINT_FILE = "fingerprint.txt"
EPISODE_FILE_GLOB = "episodes_worker*.jsonl"
MAX_CONSECUTIVE_ERRORS = 3
SERVER_WAIT_TIMEOUT_S = 15 * 60

REQUIRED_SPEC_KEYS = (
    "eval_name",
    "config_fingerprint",
    "checkpoint_config",
    "checkpoint_step",
    "task_suite",
    "replan_steps",
    "seed",
    "max_steps",
    "num_workers",
    "save_videos",
    "trials_per_variant",
    "shard_index",
    "result_dir",
    "variants",
)


def load_shard_spec(shard_file: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(shard_file).read_text())


def validate_spec(spec: dict) -> None:
    """Fail before touching the result dir if the shard file is missing anything we read."""
    missing = [key for key in REQUIRED_SPEC_KEYS if key not in spec]
    if missing:
        raise SystemExit(f"Shard spec is missing required keys: {missing}")


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
        if not existing:
            raise SystemExit(f"Corrupt fingerprint file {path}: empty. Delete it if the result dir is trustworthy.")
        if existing != fingerprint:
            raise SystemExit(
                f"Fingerprint mismatch in {result_dir}:\n"
                f"  existing results: {existing}\n"
                f"  this shard file:  {fingerprint}\n"
                "The manifest changed since those episodes were collected. Point at a fresh result dir, "
                "or regenerate with generate_manifest.py --force --discard-results."
            )
        return
    # Racing workers write identical content, so write to a private temp file and rename.
    handle, tmp_name = tempfile.mkstemp(dir=str(result_dir), prefix=FINGERPRINT_FILE, suffix=".tmp")
    with os.fdopen(handle, "w") as tmp_file:
        tmp_file.write(fingerprint + "\n")
    os.replace(tmp_name, str(path))


def load_done_episodes(result_dir: pathlib.Path) -> set:
    """Collect (variant_name, trial_idx) from every worker's file: worker counts may change between runs."""
    done = set()
    for path in sorted(pathlib.Path(result_dir).glob(EPISODE_FILE_GLOB)):
        with path.open() as episode_file:
            for line_number, raw_line in enumerate(episode_file, 1):
                line = raw_line.strip()
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


def pending_episodes(variants: list, trials_per_variant: int, done_episodes: set) -> list:
    """(variant, trial indices still to run) for every variant with outstanding trials."""
    pending = []
    for variant in variants:
        trials = [t for t in range(trials_per_variant) if (variant["name"], t) not in done_episodes]
        if trials:
            pending.append((variant, trials))
    return pending


def episodes_file_name(worker_id: int, run_id: str) -> str:
    """One file per (worker, run): overlapping launches must never append to the same file."""
    return f"episodes_worker{worker_id:02d}_{run_id}.jsonl"


def current_run_id() -> str:
    return os.environ.get("SLURM_JOB_ID") or str(os.getpid())


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


def should_save_video(save_videos: str, trial_idx: int, *, success: bool) -> bool:
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
    num_policy_steps: int,
    wall_time_s: float,
    *,
    success: bool,
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
        "max_steps": spec["max_steps"],  # lets the aggregator tell a timeout from an early stop
        "wall_time_s": round(wall_time_s, 2),
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": spec["config_fingerprint"],
    }


def append_record(episodes_path: pathlib.Path, record: dict) -> None:
    with episodes_path.open("a") as episode_file:
        episode_file.write(json.dumps(record) + "\n")
        episode_file.flush()


def connect_policy_client(host: str, port: int, timeout_s: int = SERVER_WAIT_TIMEOUT_S):
    """Wait for the policy server, but bound the wait: a launcher bug must fail, not burn the walltime."""
    import signal

    from openpi_client import websocket_client_policy

    def on_timeout(signum, frame):
        raise TimeoutError

    previous_handler = signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(timeout_s)
    try:
        return websocket_client_policy.WebsocketClientPolicy(host, port)
    except TimeoutError:
        raise SystemExit(f"No policy server at {host}:{port} after {timeout_s // 60} min; giving up.") from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_worker(
    spec: dict,
    result_dir: pathlib.Path,
    pending: list,
    completed: int,
    owned_episodes: int,
    worker_id: int,
    host: str,
    port: int,
) -> None:
    """Evaluate this worker's pending episodes; heavy imports happen here, not at module import."""
    sys.path.insert(0, str(LIBERO_EXAMPLE_DIR))
    import imageio
    from libero.libero import benchmark
    import numpy as np

    from episode_runner import LIBERO_ENV_RESOLUTION
    from episode_runner import make_libero_env
    from episode_runner import run_episode

    save_videos = spec["save_videos"]
    episodes_path = result_dir / episodes_file_name(worker_id, current_run_id())
    video_dir = result_dir / "videos"

    task_suite = benchmark.get_benchmark_dict()[spec["task_suite"]]()
    name_to_task_id = {task_suite.get_task(i).name: i for i in range(task_suite.n_tasks)}
    unknown = [variant["name"] for variant, _ in pending if variant["name"] not in name_to_task_id]
    if unknown:
        raise SystemExit(f"{len(unknown)} variants are not in task suite '{spec['task_suite']}': {unknown}")

    client = connect_policy_client(host, port)

    episodes_run, successes_run, consecutive_errors = 0, 0, 0
    for variant, pending_trials in pending:
        task_id = name_to_task_id[variant["name"]]
        init_states = task_suite.get_task_init_states(task_id)
        env, task_description = make_libero_env(task_suite.get_task(task_id), LIBERO_ENV_RESOLUTION, spec["seed"])

        try:
            for trial_idx in pending_trials:
                init_state_idx = init_state_index(trial_idx, len(init_states))
                # Per-trial seed: env.seed() reseeds the global numpy RNG that the perturbations draw
                # from, so a resumed trial sees the same stream it would have seen in one long run.
                env.seed(spec["seed"] + trial_idx)
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

                if result.error is not None:
                    # Infrastructure failure, not a task failure: leave it unrecorded so a retry reruns it.
                    consecutive_errors += 1
                    logging.error(
                        "w{} {} t{} aborted after {:.1f}s ({}); not recorded, {} in a row".format(
                            worker_id, variant["name"], trial_idx, wall_time_s, result.error, consecutive_errors
                        )
                    )
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise SystemExit(
                            f"w{worker_id}: {consecutive_errors} consecutive failed episodes "
                            f"(last: {result.error}); aborting instead of recording them as task failures."
                        )
                    continue
                consecutive_errors = 0

                # Record first: the JSONL line is authoritative, the video is a debugging aid.
                append_record(
                    episodes_path,
                    episode_record(
                        spec,
                        worker_id,
                        variant,
                        trial_idx,
                        init_state_idx,
                        result.num_policy_steps,
                        wall_time_s,
                        success=result.success,
                    ),
                )
                if should_save_video(save_videos, trial_idx, success=result.success):
                    video_dir.mkdir(parents=True, exist_ok=True)
                    suffix = "success" if result.success else "failure"
                    imageio.mimwrite(
                        video_dir / f"{variant['name']}_t{trial_idx}_{suffix}.mp4",
                        [np.asarray(image) for image in result.replay_images],
                        fps=10,
                    )

                completed += 1
                episodes_run += 1
                successes_run += int(result.success)
                logging.info(
                    "w{} [{}/{}] {} t{} {} ({} steps, {:.1f}s) | this run: {}/{} ({:.1f}%)".format(
                        worker_id,
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

    logging.info(f"w{worker_id} done: {completed}/{owned_episodes} owned episodes recorded ({episodes_run} here)")


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
    validate_spec(spec)
    if args.num_workers != spec["num_workers"]:
        logging.warning(f"Launched with {args.num_workers} workers, but the spec plans for {spec['num_workers']}.")

    result_dir = resolve_result_dir(spec, args.repo_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    check_fingerprint(result_dir, spec["config_fingerprint"])
    logging.info(
        "Shard {} of eval '{}' ({}@{}) -> {}".format(
            spec["shard_index"], spec["eval_name"], spec["checkpoint_config"], spec["checkpoint_step"], result_dir
        )
    )

    variants = owned_variants(spec["variants"], args.worker_id, args.num_workers)
    trials_per_variant = spec["trials_per_variant"]
    pending = pending_episodes(variants, trials_per_variant, load_done_episodes(result_dir))
    owned_episodes = len(variants) * trials_per_variant
    completed = owned_episodes - sum(len(trials) for _, trials in pending)
    logging.info(f"w{args.worker_id}: {len(variants)} variants, {completed}/{owned_episodes} episodes already done")
    if not pending:
        logging.info(f"w{args.worker_id}: nothing to do")
        return

    run_worker(spec, result_dir, pending, completed, owned_episodes, args.worker_id, args.host, args.port)


if __name__ == "__main__":
    main()

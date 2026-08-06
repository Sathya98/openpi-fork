#!/usr/bin/env python3
"""Expand a LIBERO-plus eval config into a manifest and per-checkpoint work shards.

Reads a config (see configs/) selecting checkpoints x perturbations x difficulties x
base tasks, resolves it against the benchmark's task_classification.json and the
checkpoints on disk, and writes shard files that the shard runner consumes.

Stdlib only: runs under any python3.8+, no libero install needed.
"""

import argparse
import collections
import datetime
import json
import pathlib
import re
import shutil
from typing import List, Optional

from perturbations import PERTURBATION_CATEGORIES

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results" / "libero_plus_system"
DEFAULT_CLASSIFICATION_JSON = "/gpfs/work5/0/prjs2050/LIBERO-plus/libero/libero/benchmark/task_classification.json"

# Variant names look like pick_up_the_<object>_and_place_it_in_the_basket_<suffix>.
BASE_TASK_PATTERN = re.compile(r"pick_up_the_(.+?)_and_place_it_in_the_basket")


def resolve_checkpoint_step(config_name: str, step: Optional[int]) -> int:
    """Return the requested step, or the latest one if step is null; verify params exist."""
    checkpoint_root = REPO_ROOT / "checkpoints" / config_name / "main"
    if not checkpoint_root.is_dir():
        raise SystemExit(f"No checkpoint directory: {checkpoint_root}")
    if step is None:
        steps = sorted(int(d.name) for d in checkpoint_root.iterdir() if d.is_dir() and d.name.isdigit())
        if not steps:
            raise SystemExit(f"No numeric step directories under {checkpoint_root}")
        step = steps[-1]
    params_dir = checkpoint_root / str(step) / "params"
    if not params_dir.is_dir():
        raise SystemExit(f"Checkpoint params not found: {params_dir}")
    return step


def select_variants(
    classification_json: str,
    task_suite: str,
    perturbations: List[str],
    difficulties: List[int],
    base_tasks: Optional[List[str]],
) -> List[dict]:
    """Filter task_classification.json entries down to the variants this eval covers."""
    entries = json.loads(pathlib.Path(classification_json).read_text())[task_suite]

    categories = {PERTURBATION_CATEGORIES[key] for key in perturbations}
    difficulty_set = set(difficulties)
    base_task_set = set(base_tasks) if base_tasks is not None else None

    variants = []
    for entry in entries:
        if entry["category"] not in categories or entry["difficulty_level"] not in difficulty_set:
            continue
        match = BASE_TASK_PATTERN.match(entry["name"])
        if match is None:
            raise SystemExit("Cannot parse base task from variant name: {}".format(entry["name"]))
        base_task = match.group(1)
        if base_task_set is not None and base_task not in base_task_set:
            continue
        variants.append(
            {
                "name": entry["name"],
                "base_task": base_task,
                "category": entry["category"],
                "difficulty": entry["difficulty_level"],
            }
        )
    if not variants:
        raise SystemExit("Selection matched zero variants; check perturbations/difficulties/base_tasks.")
    return variants


def shard_variants(variants: List[dict], num_shards: int) -> List[List[dict]]:
    """Round-robin over hardest-first order so every shard gets a similar slow/fast mix."""
    ordered = sorted(variants, key=lambda v: (-v["difficulty"], v["category"], v["name"]))
    shards = [[] for _ in range(num_shards)]
    for index, variant in enumerate(ordered):
        shards[index % num_shards].append(variant)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path, help="Path to an eval config JSON.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())

    unknown = [key for key in config["perturbations"] if key not in PERTURBATION_CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown perturbation keys {unknown}; valid keys: {sorted(PERTURBATION_CATEGORIES)}")

    output_dir = RESULTS_ROOT / config["name"]
    if output_dir.exists() and not args.force:
        raise SystemExit(
            f"{output_dir} already exists; pass --force to regenerate (this may not match collected results)."
        )

    checkpoints = [
        {"config": entry["config"], "step": resolve_checkpoint_step(entry["config"], entry["step"])}
        for entry in config["checkpoints"]
    ]
    classification_json = config.get("classification_json", DEFAULT_CLASSIFICATION_JSON)
    variants = select_variants(
        classification_json,
        config["task_suite"],
        config["perturbations"],
        config["difficulties"],
        config["base_tasks"],
    )
    shards = shard_variants(variants, config["num_shards"])

    shard_dir = output_dir / "shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    for checkpoint in checkpoints:
        tag = "{}-{}".format(checkpoint["config"], checkpoint["step"])
        for shard_index, shard in enumerate(shards):
            shard_spec = {
                "eval_name": config["name"],
                "checkpoint_config": checkpoint["config"],
                "checkpoint_step": checkpoint["step"],
                "checkpoint_dir": "checkpoints/{}/main/{}".format(checkpoint["config"], checkpoint["step"]),
                "task_suite": config["task_suite"],
                "replan_steps": config["replan_steps"],
                "seed": config["seed"],
                "max_steps": config["max_steps"],
                "num_workers": config["num_workers"],
                "save_videos": config["save_videos"],
                "trials_per_variant": config["trials_per_variant"],
                "result_dir": "results/libero_plus_system/{}/{}/shard{}".format(config["name"], tag, shard_index),
                "variants": shard,
            }
            (shard_dir / f"{tag}_shard{shard_index}.json").write_text(json.dumps(shard_spec, indent=2))

    resolved_config = dict(config, checkpoints=checkpoints, classification_json=classification_json)
    episodes_per_checkpoint = len(variants) * config["trials_per_variant"]
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_config,
        "variants": variants,
        "totals": {
            "num_variants": len(variants),
            "num_shards": config["num_shards"],
            "episodes_per_checkpoint": episodes_per_checkpoint,
            "total_episodes": episodes_per_checkpoint * len(checkpoints),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Eval '{}' -> {}".format(config["name"], output_dir))
    print("Checkpoints: " + ", ".join("{}@{}".format(c["config"], c["step"]) for c in checkpoints))
    print("Variants by category:")
    category_counts = collections.Counter(v["category"] for v in variants)
    for category in sorted(category_counts):
        print(f"  {category:<22} {category_counts[category]}")
    print("Shards (variants / episodes per checkpoint):")
    for shard_index, shard in enumerate(shards):
        print("  shard{:<3} {:>5} {:>7}".format(shard_index, len(shard), len(shard) * config["trials_per_variant"]))
    print(
        f"Total: {len(variants)} variants, {episodes_per_checkpoint} episodes per checkpoint, {episodes_per_checkpoint * len(checkpoints)} episodes overall"
    )


if __name__ == "__main__":
    main()

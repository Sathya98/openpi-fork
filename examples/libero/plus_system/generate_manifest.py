#!/usr/bin/env python3
"""Expand a LIBERO-plus eval config into a manifest and per-checkpoint work shards.

Reads a config (see configs/) selecting checkpoints x perturbations x difficulties x
base tasks, resolves it against the benchmark's task_classification.json and the
checkpoints on disk, and writes shard files that the shard runner consumes.

Stdlib only: runs under any python3.8+, no libero install needed.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import re
import shutil

from perturbations import PERTURBATION_CATEGORIES

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints"
RESULTS_ROOT = REPO_ROOT / "results" / "libero_plus_system"
DEFAULT_CLASSIFICATION_JSON = "/gpfs/work5/0/prjs2050/LIBERO-plus/libero/libero/benchmark/task_classification.json"

FINGERPRINT_FIELDS = (
    "checkpoints",
    "classification_json",
    "task_suite",
    "perturbations",
    "difficulties",
    "base_tasks",
    "trials_per_variant",
    "replan_steps",
    "seed",
    "max_steps",
    "num_shards",
)

# Object-suite variant names look like pick_up_the_<object>_and_place_it_in_the_basket_<suffix>;
# the other suites do not encode a base task in the name.
BASE_TASK_PATTERN = re.compile(r"pick_up_the_(.+?)_and_place_it_in_the_basket")

# Sensor Noise variants end in _noise_<N>, and N falls in a band of ten per corruption family.
NOISE_SUFFIX_PATTERN = re.compile(r"_noise_(\d+)$")
NOISE_FAMILIES = {
    "motion_blur": (1, 10),
    "gaussian_blur": (11, 20),
    "zoom_blur": (21, 30),
    "fog": (31, 40),
    "glass_blur": (41, 50),
}


def resolve_checkpoint_step(config_name: str, step: int | None) -> int:
    """Return the requested step, or the latest one if step is null; verify params exist."""
    checkpoint_root = CHECKPOINTS_ROOT / config_name / "main"
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


def load_suite_records(classification_json: str, task_suite: str) -> list[dict]:
    """Read the classification records of one task suite."""
    path = pathlib.Path(classification_json)
    if not path.is_file():
        raise SystemExit(f"Classification file not found: {path}")
    suites = json.loads(path.read_text())
    if task_suite not in suites:
        raise SystemExit(f"Unknown task suite '{task_suite}'; {path} has: {sorted(suites)}")
    return suites[task_suite]


def parse_base_task(variant_name: str) -> str | None:
    match = BASE_TASK_PATTERN.match(variant_name)
    return match.group(1) if match else None


def select_variants(
    records: list[dict],
    perturbations: list[str],
    difficulties: list[int],
    base_tasks: list[str] | None,
) -> tuple[list[dict], int]:
    """Filter suite records down to the variants this eval covers.

    Returns the variants and the number of records dropped for a null difficulty_level.
    """
    present_difficulties = {r["difficulty_level"] for r in records if r["difficulty_level"] is not None}
    unmatched_difficulties = sorted(set(difficulties) - present_difficulties)
    if unmatched_difficulties:
        raise SystemExit(
            f"Difficulties {unmatched_difficulties} match no record in this suite; "
            f"available: {sorted(present_difficulties)}"
        )

    base_task_set = None
    if base_tasks is not None:
        present_base_tasks = {parse_base_task(r["name"]) for r in records} - {None}
        if not present_base_tasks:
            raise SystemExit("base_tasks filtering requested, but no variant name in this suite encodes a base task.")
        unmatched_base_tasks = sorted(set(base_tasks) - present_base_tasks)
        if unmatched_base_tasks:
            raise SystemExit(
                f"base_tasks {unmatched_base_tasks} match no record in this suite; "
                f"available: {sorted(present_base_tasks)}"
            )
        base_task_set = set(base_tasks)

    categories = {PERTURBATION_CATEGORIES[key] for key in perturbations}
    difficulty_set = set(difficulties)

    variants = []
    skipped_null_difficulty = 0
    for record in records:
        if record["category"] not in categories:
            continue
        if record["difficulty_level"] is None:
            skipped_null_difficulty += 1
            continue
        if record["difficulty_level"] not in difficulty_set:
            continue
        base_task = parse_base_task(record["name"])
        if base_task_set is not None and base_task not in base_task_set:
            continue
        variants.append(
            {
                "name": record["name"],
                "base_task": base_task,
                "category": record["category"],
                "difficulty": record["difficulty_level"],
            }
        )
    if not variants:
        raise SystemExit("Selection matched zero variants; check perturbations/difficulties/base_tasks.")
    return variants, skipped_null_difficulty


def normalize_excluded_families(value: object) -> list[str]:
    """Read the optional exclude_noise_families field; absent or null means exclude nothing."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(family, str) for family in value):
        raise SystemExit(f"exclude_noise_families must be a list of family names, got {value!r}")
    return value


def exclude_noise_families(variants: list[dict], families: list[str]) -> tuple[list[dict], dict[str, int]]:
    """Drop variants whose _noise_<N> suffix falls in an excluded family's band.

    Returns the surviving variants and the number excluded per requested family.
    """
    unknown = sorted(set(families) - set(NOISE_FAMILIES))
    if unknown:
        raise SystemExit(f"Unknown noise families {unknown}; valid families: {sorted(NOISE_FAMILIES)}")

    bands = {family: NOISE_FAMILIES[family] for family in families}
    kept = []
    excluded_counts = {family: 0 for family in families}
    for variant in variants:
        match = NOISE_SUFFIX_PATTERN.search(variant["name"])
        family = None
        if match:
            index = int(match.group(1))
            family = next((name for name, (low, high) in bands.items() if low <= index <= high), None)
        if family is None:
            kept.append(variant)
        else:
            excluded_counts[family] += 1
    if not kept:
        raise SystemExit(f"exclude_noise_families {sorted(families)} excluded every selected variant.")
    return kept, excluded_counts


def shard_variants(variants: list[dict], num_shards: int) -> list[list[dict]]:
    """Round-robin over hardest-first order so every shard gets a similar slow/fast mix."""
    ordered = sorted(variants, key=lambda v: (-v["difficulty"], v["category"], v["name"]))
    shards: list[list[dict]] = [[] for _ in range(num_shards)]
    for index, variant in enumerate(ordered):
        shards[index % num_shards].append(variant)
    return shards


def compute_fingerprint(resolved_config: dict, variants: list[dict]) -> str:
    """Identity of this manifest: the runner refuses result dirs whose fingerprint differs.

    Only fields that determine the episodes count; retuning num_workers or save_videos
    must not invalidate a half-collected sweep. num_shards is included because resharding
    moves variants between result dirs, which is exactly the double-counting hazard.
    """
    payload_config = {field: resolved_config[field] for field in FINGERPRINT_FIELDS}
    # exclude_noise_families is result-determining, so it enters the payload whenever it is
    # non-empty (whether or not it happened to drop a variant). An absent or empty field is
    # left out entirely, so configs that never set it keep the fingerprint they were first
    # generated under and their already collected episodes still verify.
    if resolved_config.get("exclude_noise_families"):
        payload_config["exclude_noise_families"] = sorted(resolved_config["exclude_noise_families"])
    payload = json.dumps(
        {
            "config": payload_config,
            "variants": sorted(v["name"] for v in variants),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_checkpoints(entries: list[dict]) -> list[dict]:
    checkpoints = [
        {"config": entry["config"], "step": resolve_checkpoint_step(entry["config"], entry["step"])}
        for entry in entries
    ]
    counts = collections.Counter((c["config"], c["step"]) for c in checkpoints)
    duplicates = sorted(f"{name}@{step}" for (name, step), count in counts.items() if count > 1)
    if duplicates:
        raise SystemExit(f"Config lists the same checkpoint more than once: {duplicates}")
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path, help="Path to an eval config JSON.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    parser.add_argument(
        "--discard-results",
        action="store_true",
        help="With --force: delete already collected episodes and their result directories.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())

    unknown = [key for key in config["perturbations"] if key not in PERTURBATION_CATEGORIES]
    if unknown:
        raise SystemExit(f"Unknown perturbation keys {unknown}; valid keys: {sorted(PERTURBATION_CATEGORIES)}")
    requested_families = normalize_excluded_families(config.get("exclude_noise_families"))

    output_dir = RESULTS_ROOT / config["name"]
    if output_dir.exists() and not args.force:
        raise SystemExit(
            f"{output_dir} already exists; pass --force to regenerate (this may not match collected results)."
        )
    collected = sorted(output_dir.glob("*/shard*/episodes_*.jsonl")) if output_dir.exists() else []
    if collected and not args.discard_results:
        raise SystemExit(
            f"{output_dir} holds collected episodes ({len(collected)} files, e.g. "
            f"{collected[0].relative_to(REPO_ROOT)}). Resharding would orphan those episodes and double-count "
            "variants that move to another shard. Pass --discard-results to delete them and regenerate."
        )

    checkpoints = resolve_checkpoints(config["checkpoints"])
    classification_json = config.get("classification_json", DEFAULT_CLASSIFICATION_JSON)
    records = load_suite_records(classification_json, config["task_suite"])
    variants, skipped_null_difficulty = select_variants(
        records, config["perturbations"], config["difficulties"], config["base_tasks"]
    )
    excluded_counts = {}
    if requested_families:
        variants, excluded_counts = exclude_noise_families(variants, requested_families)

    num_shards = config["num_shards"]
    if num_shards < 1:
        raise SystemExit(f"num_shards must be at least 1, got {num_shards}")
    if num_shards > len(variants):
        print(f"WARNING: num_shards {num_shards} exceeds the {len(variants)} selected variants; using {len(variants)}.")
        num_shards = len(variants)
    shards = shard_variants(variants, num_shards)

    resolved_config = dict(
        config,
        checkpoints=checkpoints,
        classification_json=classification_json,
        num_shards=num_shards,
    )
    fingerprint = compute_fingerprint(resolved_config, variants)

    shard_dir = output_dir / "shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    if args.discard_results:
        for result_dir in sorted(p for p in output_dir.glob("*") if p.is_dir()):
            print(f"Discarding {result_dir.relative_to(REPO_ROOT)}")
            shutil.rmtree(result_dir)
    shard_dir.mkdir(parents=True)

    for checkpoint in checkpoints:
        tag = "{}-{}".format(checkpoint["config"], checkpoint["step"])
        checkpoint_dir = CHECKPOINTS_ROOT / checkpoint["config"] / "main" / str(checkpoint["step"])
        for shard_index, shard in enumerate(shards):
            shard_name = f"shard{shard_index:02d}"
            shard_spec = {
                "eval_name": config["name"],
                "config_fingerprint": fingerprint,
                "checkpoint_config": checkpoint["config"],
                "checkpoint_step": checkpoint["step"],
                "checkpoint_dir": str(checkpoint_dir.relative_to(REPO_ROOT)),
                "task_suite": config["task_suite"],
                "replan_steps": config["replan_steps"],
                "seed": config["seed"],
                "max_steps": config["max_steps"],
                "num_workers": config["num_workers"],
                "save_videos": config["save_videos"],
                "trials_per_variant": config["trials_per_variant"],
                "shard_index": shard_index,
                "num_shards": num_shards,
                "result_dir": str((output_dir / tag / shard_name).relative_to(REPO_ROOT)),
                "variants": shard,
            }
            (shard_dir / f"{tag}_{shard_name}.json").write_text(json.dumps(shard_spec, indent=2))

    episodes_per_checkpoint = len(variants) * config["trials_per_variant"]
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config_fingerprint": fingerprint,
        "config": resolved_config,
        "variants": variants,
        "totals": {
            "num_variants": len(variants),
            "num_shards": num_shards,
            "episodes_per_checkpoint": episodes_per_checkpoint,
            "total_episodes": episodes_per_checkpoint * len(checkpoints),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Eval '{}' -> {}".format(config["name"], output_dir))
    print(f"Fingerprint: {fingerprint[:16]}")
    print("Checkpoints: " + ", ".join("{}@{}".format(c["config"], c["step"]) for c in checkpoints))
    print("Variants by category:")
    category_counts = collections.Counter(v["category"] for v in variants)
    for perturbation in sorted(config["perturbations"]):
        category = PERTURBATION_CATEGORIES[perturbation]
        print(f"  {category:<22} {category_counts[category]}")
    if requested_families:
        print("Excluded noise families (variants dropped):")
        for family in sorted(excluded_counts):
            print(f"  {family:<22} {excluded_counts[family]}")
    if skipped_null_difficulty:
        print(f"Skipped {skipped_null_difficulty} records with a null difficulty_level.")
    print("Shards (variants / episodes per checkpoint):")
    for shard_index, shard in enumerate(shards):
        print("  shard{:<3} {:>5} {:>7}".format(shard_index, len(shard), len(shard) * config["trials_per_variant"]))
    print(
        f"Total: {len(variants)} variants, {episodes_per_checkpoint} episodes per checkpoint, "
        f"{episodes_per_checkpoint * len(checkpoints)} episodes overall"
    )


if __name__ == "__main__":
    main()

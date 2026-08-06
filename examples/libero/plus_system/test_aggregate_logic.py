#!/usr/bin/env python3
"""Checks for the results aggregator, on a synthetic eval directory (no libero, no GPU).

Run: python3 test_aggregate_logic.py   -> runs every check, exits 1 if any failed.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import aggregate_results

FINGERPRINT = "a" * 64
STALE_FINGERPRINT = "b" * 64

CHECKPOINTS = [{"config": "ckptA", "step": 100}, {"config": "ckptB", "step": 200}]

# shard00 gets alpha+gamma, shard01 beta+delta; 2 trials each -> 4 episodes per shard,
# 8 per checkpoint, 16 overall. delta has no base task (non-object-suite style variant).
VARIANTS = {
    "v_alpha": {"name": "v_alpha", "base_task": "ketchup", "category": "Light", "difficulty": 1},
    "v_beta": {"name": "v_beta", "base_task": "ketchup", "category": "Light", "difficulty": 2},
    "v_gamma": {"name": "v_gamma", "base_task": "milk", "category": "Camera", "difficulty": 1},
    "v_delta": {"name": "v_delta", "base_task": None, "category": "Camera", "difficulty": 2},
}
SHARD_VARIANTS = {0: ["v_alpha", "v_gamma"], 1: ["v_beta", "v_delta"]}
TRIALS_PER_VARIANT = 2


def record(
    variant_name,
    trial_idx,
    *,
    success,
    checkpoint=CHECKPOINTS[0],
    shard_index=0,
    finished_at="2026-01-01T00:00:00+00:00",
    fingerprint=FINGERPRINT,
):
    variant = VARIANTS[variant_name]
    return {
        "eval_name": "synthetic",
        "checkpoint_config": checkpoint["config"],
        "checkpoint_step": checkpoint["step"],
        "task_suite": "libero_object",
        "shard_index": shard_index,
        "worker_id": 0,
        "variant_name": variant_name,
        "base_task": variant["base_task"],
        "category": variant["category"],
        "difficulty": variant["difficulty"],
        "trial_idx": trial_idx,
        "init_state_idx": trial_idx,
        "success": success,
        "num_policy_steps": 42,
        "max_steps": 280,
        "wall_time_s": 1.0,
        "finished_at": finished_at,
        "config_fingerprint": fingerprint,
    }


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def as_line(mapping):
    return json.dumps(mapping) + "\n"


def build_eval_dir(root, *, with_episodes=True):
    """A manifest, its shard specs and (optionally) the episode files described below.

    Deduped valid episodes, all on ckptA (ckptB has nothing recorded yet):
        v_alpha t0 False, v_alpha t1 True, v_beta t0 True,
        v_gamma t0 True, v_delta t0 False, v_delta t1 False
    """
    eval_dir = pathlib.Path(root) / "synthetic"
    variants = [VARIANTS[name] for index in sorted(SHARD_VARIANTS) for name in SHARD_VARIANTS[index]]
    manifest = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "config_fingerprint": FINGERPRINT,
        "config": {
            "name": "synthetic",
            "checkpoints": CHECKPOINTS,
            "task_suite": "libero_object",
            "trials_per_variant": TRIALS_PER_VARIANT,
            "num_shards": len(SHARD_VARIANTS),
        },
        "variants": variants,
        "totals": {
            "num_variants": len(variants),
            "num_shards": len(SHARD_VARIANTS),
            "episodes_per_checkpoint": len(variants) * TRIALS_PER_VARIANT,
            "total_episodes": len(variants) * TRIALS_PER_VARIANT * len(CHECKPOINTS),
        },
    }
    (eval_dir / "shards").mkdir(parents=True)
    (eval_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for checkpoint in CHECKPOINTS:
        tag = aggregate_results.checkpoint_tag(checkpoint)
        for shard_index, names in SHARD_VARIANTS.items():
            spec = {
                "eval_name": "synthetic",
                "config_fingerprint": FINGERPRINT,
                "checkpoint_config": checkpoint["config"],
                "checkpoint_step": checkpoint["step"],
                "trials_per_variant": TRIALS_PER_VARIANT,
                "shard_index": shard_index,
                "num_shards": len(SHARD_VARIANTS),
                "result_dir": f"{eval_dir.name}/{tag}/shard{shard_index:02d}",
                "variants": [VARIANTS[name] for name in names],
            }
            (eval_dir / "shards" / f"{tag}_shard{shard_index:02d}.json").write_text(json.dumps(spec))

    if not with_episodes:
        return eval_dir

    # ckptA shard00: a duplicate whose latest record is written FIRST (so keeping the
    # last line instead of the latest timestamp would flip v_alpha t0), a torn line,
    # a blank line, and one record left over from an older manifest.
    write_lines(
        eval_dir / "ckptA-100" / "shard00" / "episodes_worker00_1.jsonl",
        [
            as_line(record("v_alpha", 0, success=False, finished_at="2026-01-01T00:00:09+00:00")),
            as_line(record("v_alpha", 0, success=True, finished_at="2026-01-01T00:00:03+00:00")),
            as_line(record("v_alpha", 1, success=True)),
            json.dumps(record("v_gamma", 1, success=True))[:40] + "\n",  # killed mid-append
            "\n",
            as_line(record("v_gamma", 0, success=True)),
            as_line(record("v_gamma", 1, success=True, fingerprint=STALE_FINGERPRINT)),
        ],
    )
    write_lines(
        eval_dir / "ckptA-100" / "shard01" / "episodes_worker01_1.jsonl",
        [
            as_line(record("v_beta", 0, success=True, shard_index=1)),
            as_line(record("v_delta", 0, success=False, shard_index=1)),
            as_line(record("v_delta", 1, success=False, shard_index=1)),
        ],
    )
    # Detritus from an unrelated run: same layout, not named by the manifest.
    write_lines(
        eval_dir / "ckpt-1" / "shard00" / "episodes_worker00_9.jsonl",
        [as_line(dict(record("v_alpha", 0, success=True), variant_name="v_bogus", category="Bogus"))],
    )
    return eval_dir


def loaded(eval_dir):
    manifest = aggregate_results.load_manifest(eval_dir)
    return manifest, aggregate_results.load_episodes(eval_dir, manifest)


def episode_keys(records):
    return sorted((r["variant_name"], r["trial_idx"], r["success"]) for r in records)


def test_dedupe_keeps_latest_finished_at():
    records = [
        record("v_alpha", 0, success=False, finished_at="2026-01-01T00:00:09+00:00"),
        record("v_alpha", 0, success=True, finished_at="2026-01-01T00:00:03+00:00"),
        record("v_beta", 0, success=False, finished_at="2026-01-01T00:00:01+00:00"),
        record("v_beta", 0, success=True, finished_at="2026-01-01T00:00:07+00:00"),
        # Same variant/trial on another checkpoint is a different episode, not a duplicate.
        record("v_beta", 0, success=False, checkpoint=CHECKPOINTS[1]),
    ]
    deduped, duplicates = aggregate_results.dedupe_records(records)
    assert duplicates == 2, duplicates
    kept = {(r["checkpoint_config"], r["variant_name"], r["trial_idx"]): r for r in deduped}
    assert len(kept) == 3, kept
    assert kept[("ckptA", "v_alpha", 0)]["success"] is False
    assert kept[("ckptA", "v_alpha", 0)]["finished_at"] == "2026-01-01T00:00:09+00:00"
    assert kept[("ckptA", "v_beta", 0)]["success"] is True
    assert kept[("ckptB", "v_beta", 0)]["success"] is False


def test_load_episodes_counts_and_content():
    with tempfile.TemporaryDirectory() as tmp:
        _, load = loaded(build_eval_dir(tmp))
        assert load.malformed_lines == 1, load.malformed_lines  # the torn line only; blanks do not count
        assert load.stale_records == 1, load.stale_records
        assert load.duplicate_records == 1, load.duplicate_records
        assert len(load.records) == 6, episode_keys(load.records)
        assert episode_keys(load.records) == [
            ("v_alpha", 0, False),
            ("v_alpha", 1, True),
            ("v_beta", 0, True),
            ("v_delta", 0, False),
            ("v_delta", 1, False),
            ("v_gamma", 0, True),
        ], episode_keys(load.records)


def test_stale_fingerprint_records_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        _, load = loaded(build_eval_dir(tmp))
        # The stale line is v_gamma t1; excluding it is what keeps Camera/difficulty 1 at n=1.
        assert all(r["config_fingerprint"] == FINGERPRINT for r in load.records)
        assert ("v_gamma", 1) not in {(r["variant_name"], r["trial_idx"]) for r in load.records}


def test_torn_line_skipped_and_counted():
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = build_eval_dir(tmp)
        path = eval_dir / "ckptA-100" / "shard00" / "episodes_worker00_1.jsonl"
        records, malformed = aggregate_results.parse_episode_file(path)
        assert malformed == 1, malformed
        assert len(records) == 5, len(records)  # 3 valid + the duplicate + the stale one

        # A line that parses but is not an episode record is malformed too.
        write_lines(path, [as_line({"hello": "world"}), as_line(record("v_alpha", 0, success=True))])
        records, malformed = aggregate_results.parse_episode_file(path)
        assert (len(records), malformed) == (1, 1), (len(records), malformed)


def test_unrelated_checkpoint_dir_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = build_eval_dir(tmp)
        manifest = aggregate_results.load_manifest(eval_dir)
        tags = [aggregate_results.checkpoint_tag(c) for c in manifest["config"]["checkpoints"]]
        files = aggregate_results.episode_files(eval_dir, tags)
        assert [p.parent.parent.name for p in files] == ["ckptA-100", "ckptA-100"], files

        load = aggregate_results.load_episodes(eval_dir, manifest)
        assert not any(r["category"] == "Bogus" for r in load.records)
        assert len(load.records) == 6, len(load.records)


def test_completion_rows():
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = build_eval_dir(tmp)
        manifest, load = loaded(eval_dir)
        rows = aggregate_results.completion_rows(eval_dir, manifest, load.records)
        assert rows == [
            {"checkpoint": "ckptA-100", "shard_index": 0, "recorded": 3, "expected": 4},
            {"checkpoint": "ckptA-100", "shard_index": 1, "recorded": 3, "expected": 4},
            # A checkpoint nobody has run yet stays in the report at 0/expected.
            {"checkpoint": "ckptB-200", "shard_index": 0, "recorded": 0, "expected": 4},
            {"checkpoint": "ckptB-200", "shard_index": 1, "recorded": 0, "expected": 4},
        ], rows
        assert sum(row["recorded"] for row in rows) == 6
        assert sum(row["expected"] for row in rows) == 16


def rate_of(summary, **cell):
    """The (episodes, success_rate) of the single row matching the given key columns."""
    mask = None
    for column, value in cell.items():
        match = summary[column].isna() if value is None else summary[column] == value
        mask = match if mask is None else (mask & match)
    rows = summary[mask]
    assert len(rows) == 1, (cell, rows)
    return int(rows["episodes"].iloc[0]), float(rows["success_rate"].iloc[0])


def close(actual, expected):
    return abs(actual - expected) < 1e-9


def test_pivot_numbers():
    with tempfile.TemporaryDirectory() as tmp:
        _, load = loaded(build_eval_dir(tmp))
        frame = aggregate_results.build_frame(load.records)
        assert set(frame["checkpoint"]) == {"ckptA-100"}, set(frame["checkpoint"])

        by_checkpoint = aggregate_results.group_success(frame, ["checkpoint", "category"])
        # Light: alpha t0 F, alpha t1 T, beta t0 T -> 2/3. Camera: gamma T, delta F, delta F -> 1/3.
        assert rate_of(by_checkpoint, checkpoint="ckptA-100", category="Light")[0] == 3
        assert close(rate_of(by_checkpoint, checkpoint="ckptA-100", category="Light")[1], 2 / 3)
        assert rate_of(by_checkpoint, checkpoint="ckptA-100", category="Camera")[0] == 3
        assert close(rate_of(by_checkpoint, checkpoint="ckptA-100", category="Camera")[1], 1 / 3)
        assert int(by_checkpoint["successes"].sum()) == 3, by_checkpoint

        by_difficulty = aggregate_results.group_success(frame, ["category", "difficulty"])
        assert rate_of(by_difficulty, category="Light", difficulty=1) == (2, 0.5)
        assert rate_of(by_difficulty, category="Light", difficulty=2) == (1, 1.0)
        assert rate_of(by_difficulty, category="Camera", difficulty=1) == (1, 1.0)
        assert rate_of(by_difficulty, category="Camera", difficulty=2) == (2, 0.0)

        # base_task None (non-object suites) must survive the grouping, not silently vanish.
        by_base_task = aggregate_results.group_success(frame, ["base_task", "category"])
        assert rate_of(by_base_task, base_task="milk", category="Camera") == (1, 1.0)
        assert rate_of(by_base_task, base_task=None, category="Camera") == (2, 0.0)
        assert close(rate_of(by_base_task, base_task="ketchup", category="Light")[1], 2 / 3)
        assert int(by_base_task["episodes"].sum()) == 6, by_base_task

        matrix = aggregate_results.format_matrix(by_difficulty, "category", "difficulty")
        assert matrix.at["Camera", 2] == "0.000 (2)", matrix
        assert matrix.at["Light", 2] == "1.000 (1)", matrix


def run_cli(eval_dir, *extra):
    return subprocess.run(
        [sys.executable, aggregate_results.__file__, str(eval_dir), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_writes_tables_and_pivots():
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = build_eval_dir(tmp)
        completed = run_cli(eval_dir)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "6 / 16" in completed.stdout, completed.stdout
        assert "0.667 (3)" in completed.stdout, completed.stdout

        csv_lines = (eval_dir / "all_episodes.csv").read_text().splitlines()
        assert len(csv_lines) == 7, csv_lines  # header + 6 episodes
        written = sorted(p.name for p in (eval_dir / "pivots").glob("*.csv"))
        assert written == [
            "base_task_by_category.csv",
            "category_by_difficulty.csv",
            "category_by_difficulty_per_checkpoint.csv",
            "checkpoint_by_category.csv",
        ], written
        per_checkpoint = (eval_dir / "pivots" / "category_by_difficulty_per_checkpoint.csv").read_text()
        assert per_checkpoint.splitlines()[0] == "checkpoint,category,difficulty,episodes,successes,success_rate"
        # The base_task pivot keeps the missing base task under a readable label.
        assert aggregate_results.NO_BASE_TASK in (eval_dir / "pivots" / "base_task_by_category.csv").read_text()

        # --no-parquet must leave no parquet behind (with an engine installed it otherwise would).
        (eval_dir / "all_episodes.parquet").unlink(missing_ok=True)
        assert run_cli(eval_dir, "--no-parquet").returncode == 0
        assert not (eval_dir / "all_episodes.parquet").exists()


def test_cli_without_records_exits_1():
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = build_eval_dir(tmp, with_episodes=False)
        completed = run_cli(eval_dir)
        assert completed.returncode == 1, completed.stdout
        assert "No episode records found" in completed.stdout, completed.stdout
        assert "0 / 16" in completed.stdout, completed.stdout
        assert not (eval_dir / "all_episodes.csv").exists()


def test_eval_dir_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = build_eval_dir(tmp, with_episodes=False)
        assert aggregate_results.resolve_eval_dir(str(eval_dir), "results/libero_plus_system") == eval_dir.resolve()
        assert aggregate_results.resolve_eval_dir("synthetic", tmp) == eval_dir.resolve()
        # A relative results root is taken from the repo root, not the working directory.
        assert aggregate_results.resolve_eval_dir("nope", "results/libero_plus_system") == (
            aggregate_results.REPO_ROOT / "results/libero_plus_system/nope"
        )


def test_missing_manifest_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            aggregate_results.load_manifest(pathlib.Path(tmp))
        except SystemExit as error:
            assert "manifest.json" in str(error), str(error)
        else:
            raise AssertionError("a directory without a manifest was accepted")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as error:  # report every failing check, not just the first
            failures.append((test.__name__, error))
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {test.__name__}")
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

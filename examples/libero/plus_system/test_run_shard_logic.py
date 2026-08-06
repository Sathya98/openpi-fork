#!/usr/bin/env python3
"""Checks for run_shard's bookkeeping logic (no libero, no GPU, no policy server).

Run: python3 test_run_shard_logic.py   -> exits 0 on success, 1 on the first failures.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import run_shard

SPEC = {
    "eval_name": "calibration",
    "config_fingerprint": "f" * 64,
    "checkpoint_config": "pi05_drema_libero_full_b32_15k",
    "checkpoint_step": 14999,
    "task_suite": "libero_object",
    "replan_steps": 8,
    "seed": 7,
    "max_steps": 280,
    "num_workers": 4,
    "save_videos": "none",
    "trials_per_variant": 3,
    "shard_index": 0,
    "num_shards": 1,
    "result_dir": "results/libero_plus_system/calibration/ckpt-1/shard00",
    "variants": [
        {"name": f"variant_{i}", "base_task": "ketchup", "category": "Light Conditions", "difficulty": 1 + i % 2}
        for i in range(7)
    ],
}


def write_episodes(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_done_set_from_all_workers():
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = pathlib.Path(tmp)
        write_episodes(
            result_dir / "episodes_worker00.jsonl",
            [{"variant_name": "variant_0", "trial_idx": 0}, {"variant_name": "variant_0", "trial_idx": 1}],
        )
        write_episodes(result_dir / "episodes_worker03.jsonl", [{"variant_name": "variant_4", "trial_idx": 2}])
        # A file from an unrelated glob must be ignored.
        (result_dir / "summary.jsonl").write_text(json.dumps({"variant_name": "variant_9", "trial_idx": 0}) + "\n")

        done = run_shard.load_done_episodes(result_dir)
        assert done == {("variant_0", 0), ("variant_0", 1), ("variant_4", 2)}, done


def test_torn_final_line_is_tolerated():
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = pathlib.Path(tmp)
        path = result_dir / "episodes_worker01.jsonl"
        good = json.dumps({"variant_name": "variant_1", "trial_idx": 0}) + "\n"
        torn = json.dumps({"variant_name": "variant_1", "trial_idx": 1})[:20]
        path.write_text(good + torn)

        assert run_shard.load_done_episodes(result_dir) == {("variant_1", 0)}

        # Torn line in the middle of a file: later lines still count.
        path.write_text(good + torn + "\n" + json.dumps({"variant_name": "variant_2", "trial_idx": 0}) + "\n")
        assert run_shard.load_done_episodes(result_dir) == {("variant_1", 0), ("variant_2", 0)}


def test_worker_split_is_a_partition():
    variants = SPEC["variants"]
    for num_workers in (1, 3, 4):
        owned = [run_shard.owned_variants(variants, worker_id, num_workers) for worker_id in range(num_workers)]
        names = [v["name"] for chunk in owned for v in chunk]
        assert sorted(names) == sorted(v["name"] for v in variants), (num_workers, names)
        assert len(names) == len(set(names)), (num_workers, names)  # disjoint
        sizes = sorted(len(chunk) for chunk in owned)
        assert sizes[-1] - sizes[0] <= 1, (num_workers, sizes)  # balanced


def test_worker_split_beyond_variant_count():
    # More workers than variants: the extra workers own nothing and must not duplicate work.
    owned = [run_shard.owned_variants(SPEC["variants"][:2], worker_id, 4) for worker_id in range(4)]
    assert [len(chunk) for chunk in owned] == [1, 1, 0, 0], owned


def test_init_state_index_wraps():
    # Layout variants have exactly one init state; every trial must reuse index 0.
    assert [run_shard.init_state_index(t, 1) for t in range(3)] == [0, 0, 0]
    # Regular variants have 50 init states; trials index distinct ones.
    assert [run_shard.init_state_index(t, 50) for t in range(3)] == [0, 1, 2]
    assert run_shard.init_state_index(51, 50) == 1


def test_fingerprint_absent_is_written():
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = pathlib.Path(tmp)
        run_shard.check_fingerprint(result_dir, "abc123")
        written = (result_dir / "fingerprint.txt").read_text().strip()
        assert written == "abc123", written
        assert not list(result_dir.glob("*.tmp*")), "temp file left behind"


def test_fingerprint_match_is_accepted_and_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = pathlib.Path(tmp)
        (result_dir / "fingerprint.txt").write_text("abc123\n")
        try:
            run_shard.check_fingerprint(result_dir, "abc123")
        except SystemExit as error:
            raise AssertionError(f"matching fingerprint was refused: {error}")
        assert (result_dir / "fingerprint.txt").read_text() == "abc123\n"


def test_fingerprint_mismatch_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = pathlib.Path(tmp)
        (result_dir / "fingerprint.txt").write_text("abc123\n")
        try:
            run_shard.check_fingerprint(result_dir, "def456")
        except SystemExit as error:
            assert "abc123" in str(error) and "def456" in str(error), str(error)
        else:
            raise AssertionError("mismatched fingerprint was accepted")


def test_episode_record_fields():
    record = run_shard.episode_record(
        SPEC,
        worker_id=2,
        variant=SPEC["variants"][3],
        trial_idx=1,
        init_state_idx=0,
        success=True,
        num_policy_steps=137,
        wall_time_s=42.125,
    )
    expected_keys = {
        "eval_name",
        "checkpoint_config",
        "checkpoint_step",
        "task_suite",
        "shard_index",
        "worker_id",
        "variant_name",
        "base_task",
        "category",
        "difficulty",
        "trial_idx",
        "init_state_idx",
        "success",
        "num_policy_steps",
        "wall_time_s",
        "finished_at",
        "config_fingerprint",
    }
    assert set(record) == expected_keys, set(record) ^ expected_keys
    assert record["variant_name"] == "variant_3" and record["difficulty"] == SPEC["variants"][3]["difficulty"]
    assert record["worker_id"] == 2 and record["trial_idx"] == 1 and record["init_state_idx"] == 0
    assert record["success"] is True and record["num_policy_steps"] == 137
    assert record["config_fingerprint"] == SPEC["config_fingerprint"]
    assert record["finished_at"].endswith("+00:00"), record["finished_at"]
    # Must survive the JSONL round trip the runner performs.
    assert json.loads(json.dumps(record)) == record


def test_video_policies():
    cases = {
        "none": {(0, True): False, (0, False): False, (2, False): False},
        "failures_first_trial": {(0, True): False, (0, False): True, (1, False): False, (1, True): False},
        "all": {(0, True): True, (0, False): True, (2, False): True},
    }
    for policy, expectations in cases.items():
        for (trial_idx, success), expected in expectations.items():
            got = run_shard.should_save_video(policy, trial_idx, success)
            assert got is expected, (policy, trial_idx, success, got)

    # Frames are only kept when a video could still be produced for that trial.
    assert run_shard.should_record_images("none", 0) is False
    assert run_shard.should_record_images("failures_first_trial", 0) is True
    assert run_shard.should_record_images("failures_first_trial", 1) is False
    assert run_shard.should_record_images("all", 5) is True

    bad_calls = (
        lambda: run_shard.should_save_video("sometimes", 0, False),
        lambda: run_shard.should_record_images("sometimes", 0),
    )
    for bad_call in bad_calls:
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("unknown save_videos policy was accepted")


def test_result_dir_is_repo_relative():
    resolved = run_shard.resolve_result_dir(SPEC, pathlib.Path("/repo"))
    assert resolved == pathlib.Path("/repo/results/libero_plus_system/calibration/ckpt-1/shard00"), resolved


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

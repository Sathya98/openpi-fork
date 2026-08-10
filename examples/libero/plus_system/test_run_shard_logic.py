#!/usr/bin/env python3
"""Checks for run_shard's bookkeeping logic (no libero, no GPU, no policy server).

Run: python3 test_run_shard_logic.py   -> runs every check, exits 1 if any failed.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
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


def test_pending_episodes():
    variants = SPEC["variants"][:3]
    trials = SPEC["trials_per_variant"]

    nothing_done = run_shard.pending_episodes(variants, trials, set())
    assert [(v["name"], t) for v, t in nothing_done] == [(v["name"], [0, 1, 2]) for v in variants], nothing_done

    all_done = {(v["name"], t) for v in variants for t in range(trials)}
    assert run_shard.pending_episodes(variants, trials, all_done) == []

    partial = {("variant_0", 0), ("variant_0", 2), ("variant_2", 1)}
    assert [(v["name"], t) for v, t in run_shard.pending_episodes(variants, trials, partial)] == [
        ("variant_0", [1]),
        ("variant_1", [0, 1, 2]),
        ("variant_2", [0, 2]),
    ]

    # Episodes of variants owned by other workers must not shrink this worker's queue.
    unowned = {("variant_5", t) for t in range(trials)}
    assert run_shard.pending_episodes(variants, trials, unowned) == nothing_done


def test_append_record_round_trips_into_done_set():
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = pathlib.Path(tmp)
        episodes_path = result_dir / run_shard.episodes_file_name(3, "12345")
        for trial_idx in range(2):
            run_shard.append_record(
                episodes_path,
                run_shard.episode_record(SPEC, 3, SPEC["variants"][1], trial_idx, 0, 42, 1.0, success=True),
            )
        # The per-run file name must stay visible to the resume glob.
        assert run_shard.load_done_episodes(result_dir) == {("variant_1", 0), ("variant_1", 1)}


def test_completed_shard_needs_no_simulator():
    # Under system python3 (no libero installed) this also proves the heavy imports stay unreached.
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = pathlib.Path(tmp)
        shard_file = repo_root / "shard.json"
        shard_file.write_text(json.dumps(SPEC))
        result_dir = repo_root / SPEC["result_dir"]
        result_dir.mkdir(parents=True)
        (result_dir / "fingerprint.txt").write_text(SPEC["config_fingerprint"] + "\n")
        write_episodes(
            result_dir / run_shard.episodes_file_name(0, "99"),
            [
                {"variant_name": v["name"], "trial_idx": t}
                for v in SPEC["variants"]
                for t in range(SPEC["trials_per_variant"])
            ],
        )
        completed = subprocess.run(
            [
                sys.executable,
                run_shard.__file__,
                "--shard-file",
                str(shard_file),
                "--worker-id",
                "0",
                "--num-workers",
                "1",
                "--repo-root",
                str(repo_root),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "nothing to do" in completed.stderr, completed.stderr


def test_spec_missing_keys_is_refused():
    incomplete = {key: value for key, value in SPEC.items() if key != "max_steps"}
    try:
        run_shard.validate_spec(incomplete)
    except SystemExit as error:
        assert "max_steps" in str(error), str(error)
    else:
        raise AssertionError("incomplete shard spec was accepted")


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
            raise AssertionError(f"matching fingerprint was refused: {error}") from error
        assert (result_dir / "fingerprint.txt").read_text() == "abc123\n"


def test_fingerprint_mismatch_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = pathlib.Path(tmp)
        (result_dir / "fingerprint.txt").write_text("abc123\n")
        try:
            run_shard.check_fingerprint(result_dir, "def456")
        except SystemExit as error:
            assert "abc123" in str(error), str(error)
            assert "def456" in str(error), str(error)
        else:
            raise AssertionError("mismatched fingerprint was accepted")


def test_episode_record_fields():
    record = run_shard.episode_record(
        SPEC,
        worker_id=2,
        variant=SPEC["variants"][3],
        trial_idx=1,
        init_state_idx=0,
        num_policy_steps=137,
        wall_time_s=42.125,
        success=True,
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
        "max_steps",
        "wall_time_s",
        "finished_at",
        "config_fingerprint",
    }
    assert set(record) == expected_keys, set(record) ^ expected_keys
    assert record["variant_name"] == "variant_3"
    assert record["difficulty"] == SPEC["variants"][3]["difficulty"]
    assert record["worker_id"] == 2
    assert record["trial_idx"] == 1
    assert record["init_state_idx"] == 0
    assert record["success"] is True
    assert record["num_policy_steps"] == 137
    assert record["max_steps"] == SPEC["max_steps"]
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
            got = run_shard.should_save_video(policy, trial_idx, success=success)
            assert got is expected, (policy, trial_idx, success, got)

    # Frames are only kept when a video could still be produced for that trial.
    assert run_shard.should_record_images("none", 0) is False
    assert run_shard.should_record_images("failures_first_trial", 0) is True
    assert run_shard.should_record_images("failures_first_trial", 1) is False
    assert run_shard.should_record_images("all", 5) is True

    bad_calls = (
        lambda: run_shard.should_save_video("sometimes", 0, success=False),
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


def noise_variants(count=50):
    """One Sensor Noise variant per severity index, spanning all five families."""
    return [
        {
            "name": f"pick_up_the_ketchup_and_place_it_in_the_basket_noise_{i}",
            "base_task": "ketchup",
            "category": "Sensor Noise",
            "difficulty": 2,
        }
        for i in range(1, count + 1)
    ]


def test_deferring_a_family_drops_only_its_band():
    kept = run_shard.defer_noise_families(noise_variants(), ["zoom_blur"])
    indices = sorted(int(v["name"].rsplit("_", 1)[1]) for v in kept)
    assert indices == [i for i in range(1, 51) if not 21 <= i <= 30], indices


def test_deferring_nothing_is_a_no_op():
    variants = noise_variants()
    assert run_shard.defer_noise_families(variants, []) == variants


def test_deferring_leaves_non_noise_variants_alone():
    others = SPEC["variants"]
    kept = run_shard.defer_noise_families(others + noise_variants(), ["motion_blur", "zoom_blur"])
    assert [v for v in kept if v["category"] != "Sensor Noise"] == others


def test_deferring_every_owned_variant_yields_nothing_to_do():
    # A worker whose whole stride is deferred must return empty, not abort the shard.
    assert run_shard.defer_noise_families(noise_variants()[20:30], ["zoom_blur"]) == []


def test_unknown_family_is_rejected():
    try:
        run_shard.defer_noise_families(noise_variants(), ["zoomblur"])
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown noise family was accepted")


def test_deferral_matches_the_manifest_generator_bands():
    # run_shard must classify families exactly as generate_manifest does, or a deferred
    # run and a regenerated manifest would disagree about which variants exist.
    import generate_manifest

    for family in generate_manifest.NOISE_FAMILIES:
        deferred = run_shard.defer_noise_families(noise_variants(), [family])
        excluded, _ = generate_manifest.exclude_noise_families(noise_variants(), [family])
        assert deferred == excluded, family


def test_family_list_parses_from_comma_separated_flag():
    assert run_shard.parse_family_list("motion_blur,zoom_blur") == ["motion_blur", "zoom_blur"]
    assert run_shard.parse_family_list(" motion_blur , zoom_blur ") == ["motion_blur", "zoom_blur"]
    assert run_shard.parse_family_list("") == []
    assert run_shard.parse_family_list(None) == []


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

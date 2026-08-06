#!/usr/bin/env python3
"""Checks for the noise-family filter and its effect on the config fingerprint.

Run: python3 test_generate_manifest_logic.py   -> runs every check, exits 1 if any failed.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import generate_manifest

BASE_NAME = "pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0"

RESOLVED_CONFIG = {
    "checkpoints": [{"config": "ckptA", "step": 100}],
    "classification_json": "/somewhere/task_classification.json",
    "task_suite": "libero_object",
    "perturbations": ["noise"],
    "difficulties": [1, 2, 3],
    "base_tasks": None,
    "trials_per_variant": 50,
    "replan_steps": 8,
    "seed": 7,
    "max_steps": 280,
    "num_shards": 8,
}


def variant(name: str) -> dict:
    return {"name": name, "base_task": "alphabet_soup", "category": "Sensor Noise", "difficulty": 1}


def noise_variants() -> list[dict]:
    return [variant(f"{BASE_NAME}_noise_{index}") for index in range(1, 51)]


def test_band_boundaries():
    kept, counts = generate_manifest.exclude_noise_families(noise_variants(), ["glass_blur"])
    assert counts == {"glass_blur": 10}, counts
    assert len(kept) == 40, len(kept)
    surviving = {int(name.rsplit("_", 1)[1]) for name in (v["name"] for v in kept)}
    assert surviving == set(range(1, 41)), sorted(surviving)


def test_several_families_counted_separately():
    kept, counts = generate_manifest.exclude_noise_families(noise_variants(), ["fog", "motion_blur"])
    assert counts == {"fog": 10, "motion_blur": 10}, counts
    assert len(kept) == 30, len(kept)


def test_variants_without_a_noise_suffix_are_kept():
    others = [variant(f"{BASE_NAME}_light_43"), variant(f"{BASE_NAME}_noise_43_extra")]
    kept, counts = generate_manifest.exclude_noise_families(others + noise_variants(), ["glass_blur"])
    assert counts == {"glass_blur": 10}, counts
    assert {v["name"] for v in others} <= {v["name"] for v in kept}


def test_unknown_family_is_refused():
    try:
        generate_manifest.exclude_noise_families(noise_variants(), ["glass_blurr"])
    except SystemExit as error:
        assert "glass_blurr" in str(error), str(error)
        assert "glass_blur'" in str(error), "the message must list the valid families: " + str(error)
    else:
        raise AssertionError("an unknown noise family was accepted")


def test_excluding_everything_is_refused():
    try:
        generate_manifest.exclude_noise_families(noise_variants()[40:], ["glass_blur"])
    except SystemExit as error:
        assert "excluded every selected variant" in str(error), str(error)
    else:
        raise AssertionError("an empty selection was accepted")


def test_fingerprint_unchanged_when_field_is_absent_or_empty():
    """Configs predating the field must keep the fingerprint their episodes were collected under."""
    variants = noise_variants()
    baseline = generate_manifest.compute_fingerprint(RESOLVED_CONFIG, variants)
    empty = generate_manifest.compute_fingerprint(dict(RESOLVED_CONFIG, exclude_noise_families=[]), variants)
    assert empty == baseline, (empty, baseline)


def test_fingerprint_changes_with_the_field():
    variants = noise_variants()[:40]
    without = generate_manifest.compute_fingerprint(RESOLVED_CONFIG, variants)
    with_field = generate_manifest.compute_fingerprint(
        dict(RESOLVED_CONFIG, exclude_noise_families=["glass_blur"]), variants
    )
    assert with_field != without
    reordered = generate_manifest.compute_fingerprint(
        dict(RESOLVED_CONFIG, exclude_noise_families=["fog", "glass_blur"]), variants
    )
    same_set = generate_manifest.compute_fingerprint(
        dict(RESOLVED_CONFIG, exclude_noise_families=["glass_blur", "fog"]), variants
    )
    assert reordered == same_set, "family order must not change the fingerprint"


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

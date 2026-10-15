"""Extracted pepper episodes (images + CSVs) -> a LeRobot v2.1 dataset openpi can train on.

Mirrors examples/libero/convert_libero_data_to_lerobot.py and the schema of
pepper_var_real10ep_lrd: a 256x256 front image (the 720x720 crop at x=280 of
the 1280x720 frame, as the RLDS stores it), the same image again as
`wrist_image` (DremaInputs masks it), an 8-dim state (ee position, rotation
vector, two finger widths in metres), the 7-dim normalised action the CSV
already holds (deltas / pos_limit, rot_limit; gripper +1 open, -1 close), and
the episode's own instruction as the task.

    HF_LEROBOT_HOME=/mnt/data/datasets/pepper-in-pan/lerobot-v21 \\
    python convert_pepper_episodes_to_lerobot.py --episodes ~/rlds-episodes/pepper-in-pan \\
        --repo-id pepper_var_real18ep_lrd --include 00 01 ... 17
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from scipy.spatial.transform import Rotation

SIZE = 256


def crop_resize(image: np.ndarray, crop: dict) -> np.ndarray:
    """The RLDS crop (a 720 square at x=280 of the 1280x720 frame) on an image of
    any resolution that shares the real camera's geometry, to SIZE, RGB."""
    h, w = image.shape[:2]
    sx, sy = w / crop["original_width"], h / crop["original_height"]
    top, left, side = crop["crop_offset_y"], crop["crop_offset_x"], crop["crop_size"]
    square = image[int(round(top * sy)) : int(round((top + side) * sy)), int(round(left * sx)) : int(round((left + side) * sx))]
    return cv2.cvtColor(cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def frame_image(root: Path, index: int, crop: dict) -> np.ndarray:
    return crop_resize(cv2.imread(str(root / "images" / f"frame_{index:06d}.png")), crop)


def _gray(a: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def twin_regrid(real_root: Path, twin_dir: Path) -> tuple[list[np.ndarray], float]:
    """A DreMa replay's raw renders resampled onto the real camera's grid.

    The replay is rendered slightly off the real camera (scale ~0.98 x 0.96);
    the RLDS twins were regridded to the real camera, so this does the same
    with an affine ECC fit on frame 0 (arm at home, background dominates),
    applied to every frame. Returns the regridded frames and the ECC rho."""
    files = sorted(twin_dir.glob("*.png"), key=lambda p: int(p.stem))
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    ref = cv2.resize(cv2.imread(str(real_root / "images" / "frame_000000.png")), (w, h), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    rho, warp = cv2.findTransformECC(_gray(ref), _gray(first), warp, cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    frames = [cv2.warpAffine(cv2.imread(str(f)), warp, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,   # keeps the render's sharpness; bilinear loses a third of it
                             borderMode=cv2.BORDER_REPLICATE) for f in files]
    return frames, float(rho)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=Path, default=Path.home() / "rlds-episodes/pepper-in-pan")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--include", nargs="+", required=True, help="episode directory names")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--twins", type=Path, default=None,
                   help="directory of aligned DreMa replays (<ep>/front_rgb/*.png); adds each included episode's twin "
                        "with the real episode's labels and the render regridded to the real camera")
    args = p.parse_args()

    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        shutil.rmtree(out)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    total = 0
    for name in args.include:
        root = args.episodes / name
        meta = json.loads((root / "metadata.json").read_text())
        actions = pd.read_csv(root / "actions.csv")
        states = pd.read_csv(root / "robot_states.csv")
        assert len(actions) == len(states), (name, len(actions), len(states))
        quat = states[["ee_qx", "ee_qy", "ee_qz", "ee_qw"]].to_numpy()
        rotvec = Rotation.from_quat(quat).as_rotvec()
        sources = [("real", lambda i: frame_image(root, int(actions.frame[i]), meta["crop_info"]))]
        if args.twins is not None:
            frames, rho = twin_regrid(root, args.twins / name / "front_rgb")
            assert len(frames) == len(actions), (name, len(frames), len(actions))
            sources.append(("twin", lambda i, frames=frames: crop_resize(frames[i], meta["crop_info"])))
            print(f"  {name} twin: ECC rho {rho:.3f}")
        for label, image_of in sources:
            for i in range(len(actions)):
                image = image_of(i)
                state = np.concatenate([
                    states.loc[i, ["ee_x", "ee_y", "ee_z"]].to_numpy(dtype=np.float32),
                    rotvec[i].astype(np.float32),
                    states.loc[i, ["finger_0", "finger_1"]].to_numpy(dtype=np.float32),
                ])
                action = actions.loc[i, ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]].to_numpy(dtype=np.float32)
                dataset.add_frame({"image": image, "wrist_image": image, "state": state, "actions": action, "task": meta["instruction"]})
            dataset.save_episode()
            total += len(actions)
            print(f"  {name} {label}: {len(actions)} frames, instruction {meta['instruction']!r}")
    if hasattr(dataset, "consolidate"):
        dataset.consolidate(run_compute_stats=False)
    print(f"{args.repo_id}: {len(args.include)} episodes, {total} frames -> {out}")


if __name__ == "__main__":
    main()

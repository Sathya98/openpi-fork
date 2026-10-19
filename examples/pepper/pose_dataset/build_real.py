"""Step 1 (robot-stack venv): extracted real episodes -> a robot-stack LeRobot dataset at 5 Hz.

Action = the next measured hand pose (xyz + r6) + gripper (1 open, 0 closed): the
stored per-step deltas are exactly pose[t+1] - pose[t], so this is the same motion
as an absolute target. State = hand pose + gripper width + 7 joints.

    python build_real.py --out fr3-vislab/pepper-old3-real 05 06 14
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation as R
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from fr3_control import pose_to_vec

p = argparse.ArgumentParser()
p.add_argument("episodes", nargs="+")
p.add_argument("--src", type=Path, default=Path.home() / "rlds-episodes/pepper-in-pan")
p.add_argument("--out", required=True)
p.add_argument("--task", default="put the green bell pepper in the pan")
args = p.parse_args()
SRC, EPS, OUT = args.src, args.episodes, args.out
# Feature layout of the robot-stack recordings (FrankaJointPos).
TEMPLATE = LeRobotDataset("fr3-vislab/put-the-green-bell-pepper-in-the-pan")
TASK = args.task


root = HF_LEROBOT_HOME / OUT
assert not root.exists(), root
features = {k: dict(v) for k, v in TEMPLATE.meta.info["features"].items()
            if k in ("action", "observation.state", "observation.images.zed_left")}
ds = LeRobotDataset.create(repo_id=OUT, fps=5, root=root, robot_type=TEMPLATE.meta.robot_type,
                           features=features, image_writer_threads=4)
manifest = []
for ep in EPS:
    s = pd.read_csv(SRC / ep / "robot_states.csv")
    a = pd.read_csv(SRC / ep / "actions.csv")
    assert len(s) == len(a)
    pose = np.array([pose_to_vec(r[["ee_x", "ee_y", "ee_z"]].values.astype(float),
                                 R.from_quat(r[["ee_qx", "ee_qy", "ee_qz", "ee_qw"]].values.astype(float)))
                     for _, r in s.iterrows()], dtype=np.float32)
    grip = np.clip((s.finger_0 + s.finger_1).values / 0.08, 0, 1).astype(np.float32)
    joints = s[[f"joint_{i}" for i in range(7)]].values.astype(np.float32)
    grip_act = (a.gripper.values > 0).astype(np.float32)
    for t in range(len(s)):
        target = pose[min(t + 1, len(s) - 1)]
        img = np.asarray(Image.open(SRC / ep / "images" / f"frame_{t:06d}.png").convert("RGB"))
        assert img.shape == (720, 1280, 3), img.shape
        ds.add_frame({"action": np.array([*target, grip_act[t]], dtype=np.float32),
                      "observation.state": np.array([*pose[t], grip[t], *joints[t]], dtype=np.float32),
                      "observation.images.zed_left": img, "task": TASK})
    ds.save_episode()
    meta = json.loads((SRC / ep / "metadata.json").read_text())
    manifest.append({"episode": len(manifest), "source": str(SRC / ep), "source_episode": int(ep),
                     "bag": meta["bag_path"]})
    print(ep, len(s), flush=True)
ds.finalize()
(root / "source_bags.json").write_text(json.dumps(manifest, indent=2))
print("done", root)

"""Step 5 (openpi venv): the npz episodes -> LeRobot v2.1 for openpi (keys image/state/actions; state = pose + gripper, joints dropped)."""
import shutil, sys
from pathlib import Path
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
src, repo = Path(sys.argv[1]), sys.argv[2]
files = sorted(src.glob("*.npz")); z0 = np.load(files[0])
out = HF_LEROBOT_HOME / repo
if out.exists(): shutil.rmtree(out)
ds = LeRobotDataset.create(repo_id=repo, robot_type="franka_jointpos", fps=int(z0["fps"]), features={
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (10,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (z0["actions"].shape[1],), "names": ["actions"]},
}, image_writer_threads=8, image_writer_processes=4)
n = 0
for f in files:
    z = np.load(f)
    for im, s, a in zip(z["image"], z["state"][:, :10], z["actions"]):  # ee pose + gripper; joints dropped
        ds.add_frame({"image": im, "state": s, "actions": a, "task": str(z["task"])})
    ds.save_episode(); n += len(z["image"])
print(repo, len(files), "episodes", n, "frames ->", out)

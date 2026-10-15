"""Step 4 (robot-stack venv): a robot-stack dataset -> one npz per episode (256 px image, 17-D state, 10-D action, task)."""
import sys
from pathlib import Path
import cv2, numpy as np
from lerobot.datasets import LeRobotDataset
repo, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ds = LeRobotDataset(repo)
for ep in range(ds.meta.total_episodes):
    e = ds.meta.episodes[ep]
    imgs, st, ac = [], [], []
    for i in range(e["dataset_from_index"], e["dataset_to_index"]):
        x = ds[i]
        im = (x["observation.images.zed_left"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        imgs.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA))
        st.append(x["observation.state"].numpy()); ac.append(x["action"].numpy()); task = x["task"]
    np.savez(out / f"{ep:03d}.npz", image=np.stack(imgs), state=np.stack(st).astype(np.float32),
             actions=np.stack(ac).astype(np.float32), task=task, fps=ds.fps)
    print(ep, len(imgs), flush=True)

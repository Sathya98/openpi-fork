"""Step 2b: DreMa replays (full-size renders) -> the fr3-drema-to-dataset input layout.

The render is slightly off the real camera, so it is regridded onto it: an ECC affine
of render frame 0 onto real frame 0, Lanczos, then 1280x720. Sim state from
object_poses.pkl, one row per frame. Output dirs are numbered in episode order.

    python replays_to_frames.py [--real <extracted episodes>] <replays_dir> <out_dir> 05 06 14
"""
import pickle, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R

args = sys.argv[1:]
REAL = Path.home() / "rlds-episodes/pepper-in-pan"  # the real episodes the replays came from
if args[:1] == ["--real"]:
    REAL, args = Path(args[1]), args[2:]
SRC, OUT, EPS = Path(args[0]), Path(args[1]), args[2:]
def gray(a): return cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (5, 5), 0)

for idx, ep in enumerate(EPS):
    rep = SRC / ep / "episode0"
    files = sorted((rep / "front_rgb").glob("*.png"), key=lambda p: int(p.stem))
    poses = pickle.load(open(rep / "object_poses.pkl", "rb"))
    states = pd.read_csv(REAL / ep / "robot_states.csv")
    assert len(files) == len(poses) == len(states), (ep, len(files), len(poses), len(states))
    first = cv2.imread(str(files[0])); h, w = first.shape[:2]
    ref = cv2.resize(cv2.imread(str(REAL / ep / "images" / "frame_000000.png")), (w, h), interpolation=cv2.INTER_AREA)
    rho, warp = cv2.findTransformECC(gray(ref), gray(first), np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    out = OUT / f"{idx:02d}" / "episode0"; (out / "dataset_frames").mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(files):
        im = cv2.warpAffine(cv2.imread(str(f)), warp, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(out / "dataset_frames" / f"{k:06d}.png"), cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA))
    jp = np.array([p["robot"]["joint_positions"] for p in poses])
    names = [o["name"] for o in poses[0]["objects"]]
    stamps = states.timestamp.values.astype(float)
    np.savez(out / "dataset_frames.npz", stamps=stamps, fps=5.0, camera="front", object_names=np.array(names),
             t=stamps - stamps[0], t_rec=stamps - stamps[0], q_sim=jp[:, :7], fingers_sim=jp[:, 10:12],
             obj_pos=np.array([[o["position_world"] for o in p["objects"]] for p in poses]),
             obj_quat=np.array([[R.from_matrix(o["orientation_world"]).as_quat() for o in p["objects"]] for p in poses]))
    (out / "SOURCE").write_text(f"old-pipeline replay {rep} (ep {ep}), ECC rho {rho:.4f}, render {w}x{h}\n")
    print(idx, ep, len(files), f"{w}x{h}", f"rho={rho:.4f}", flush=True)

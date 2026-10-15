# Pepper-in-pan as a pose dataset for openpi

How `pepper_old3_05_06_14_crop_lrd` (and the 18-episode `pepper_old_all_crop_lrd`) were
built: real episodes plus their DreMa twins, LeRobot v2.1, trained with the
`pi05_pepper3_rel_*` / `pi05_pepper_old_delta_*` configs.

## How this differs from the OpenVLA-format sets

`../convert_pepper_episodes_to_lerobot.py` builds the `pepper_var_*_lrd` sets in the
OpenVLA layout: 7-D per-step deltas (xyz + rotation vector, divided by a limit,
gripper ±1), 8-D state. The episodes and the image are the same here (the 720×720
crop at x=280 of the ZED frame, at 256 px, 5 Hz); the action is not:

| | OpenVLA-format sets | these sets |
|---|---|---|
| `actions` | 7-D per-step delta | 10-D **absolute** next hand pose: xyz + r6 (two rotation-matrix columns) + gripper 1/0 |
| `state` | 8-D | 10-D, same layout as the action |
| wrist image | copy of the front image | none |

The dataset stores absolute poses. The chunk-relative representation is made at
training time by the config (`DremaRelPoseDataConfig`: xyz − xyz₀ and R_k R₀ᵀ against the
state at the chunk start) and undone at inference (`AbsolutePoseActions`). Every frame
is kept: no no-op filtering.

## Pipeline

Steps 1–4 run in the robot-stack venv (`fr3_control`, its LeRobot fork, the `fr3-*`
tools), step 5 in this repo's venv.

1. Real episodes from the extracted recordings (`~/rlds-episodes/pepper-in-pan/<ep>/`):
   `python build_real.py --out fr3-vislab/pepper-old3-real 05 06 14`, then
   `fr3-crop fr3-vislab/pepper-old3-real --out fr3-vislab/pepper-old3-real-crop512 --crop 0 280 720 720 --size 512 512`
2. DreMa twins:
   a. Replay each episode in its aligned DreMa scene with the old (settling Cartesian)
      controller, full-size renders: IVI, `Drema2-oldreplay`,
      `jobs/pick_and_place_green_bell_pepper/replay_full.sbatch`
      (`ENVS_DIR=3_drema_envs_aligned`, array = episode numbers).
   b. `python replays_to_frames.py <replays> <replays_regrid> 05 06 14`
   c. `fr3-drema-to-dataset fr3-vislab/pepper-old3-real --replays <replays_regrid> --out fr3-vislab/pepper-old3-drema`,
      then `fr3-crop` as in step 1 to `...-drema-crop512`.
3. `fr3-merge fr3-vislab/pepper-old3-real-crop512 fr3-vislab/pepper-old3-drema-crop512 --out fr3-vislab/pepper-old3-all-crop512`
4. `python v3_dump.py fr3-vislab/pepper-old3-all-crop512 <npz_dir>` (512 → 256 px)
5. `HF_LEROBOT_HOME=<dir> python v21_write.py <npz_dir> pepper_old3_05_06_14_crop_lrd`

Episode order in the result: the real episodes, then their twins in the same order.

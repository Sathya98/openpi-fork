# LIBERO-plus evaluation (self-contained client)

This directory is a **model-agnostic LIBERO-plus eval client**. It runs the perturbed
LIBERO robustness benchmark through the EEF `move_to_pose` control wrapper against any
policy served over the **openpi websocket protocol**.

It does **not** include a policy server. Bring your own:
- **pi0.5** → openpi's `scripts/serve_policy.py` (in `pi_env`)
- **any other model** → a small `serve_*_websocket.py` (see the `our-openvla-oft` port)

The websocket is the seam: everything here — perturbation selection, difficulty, trials,
the perturbed MuJoCo envs, the `move_to_pose` control loop, sharding/resume, aggregation —
is model-agnostic. The only model-specific piece is the server.

## What's in here

| Path | Role |
|---|---|
| `setup_libero_plus_env.sh` | one-shot bootstrap: clone LIBERO-plus + build `libero_plus_env` + assets + fixups |
| `episode_runner.py` | env build + one-episode rollout — the `client.infer()` seam |
| `libero_eef_control_wrapper.py`, `libero_utils.py` | the `move_to_pose` EEF control loop |
| `plus_system/generate_manifest.py` | config JSON → sharded manifest (+ fingerprint) |
| `plus_system/run_shard.py` | resumable per-worker runner (JSONL episodes, SIGKILL-safe) |
| `plus_system/aggregate_results.py` | dedupe → CSV/parquet + completion report |
| `plus_system/perturbations.py` | perturbation + difficulty selection |
| `plus_system/configs/*.json` | sweep configs |
| `main_plus.py` | single-job client (one perturbation dimension; quick runs / smoke) |

Slurm launchers live at the repo root: `slurm/eval_plus_system.slurm`, `slurm/submit_plus_system.sh`.

## 1. One-time setup (simulator + python env)

```bash
./setup_libero_plus_env.sh --prefix ../../.libero_plus --python python3.8 \
    --magick-home /path/to/imagemagick        # omit if ImageMagick is system-installed
```

Clones `github.com/sylvestf/LIBERO-plus`, builds the **py3.8** venv `libero_plus_env`,
installs deps + `openpi-client`, downloads the **~6.4 GB** `assets.zip` from HF
(`Sylvest/LIBERO-plus`), writes the LIBERO `config.yaml` + known fixups, and emits a
`plus_env.sh`. Prereqs: **python3.8**, **ImageMagick** (for `wand`), ~9 GB disk.

Knobs: `--prefix DIR` (where the clone + venv land; a *relative* prefix resolves against
this script's dir), `--python`, `--magick-home`, `--openpi-client`, `--skip-assets`,
`--assets-only`. Re-running is idempotent.

## 2. Bring a policy server (not included)

```bash
# pi0.5 — from the openpi repo, in pi_env:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/serve_policy.py \
    --port 8000 policy:checkpoint --policy.config <cfg> --policy.dir <ckpt>
```

**Server contract:** respond to `WebsocketClientPolicy.infer(obs) → {"actions": chunk}`
of shape `(replan_steps, action_dim)`, plus `/healthz`. Actions must be in the
**EEF-delta** space the wrapper expects (position + axis-angle delta + gripper).

## 3. Configure a sweep

Copy/edit a config under `plus_system/configs/`:

```json
{
  "name": "my_sweep",
  "checkpoints": [{"config": "<train-config-or-label>", "step": null}],
  "task_suite": "libero_object",
  "perturbations": ["light", "background", "noise", "layout", "camera", "robot", "language"],
  "difficulties": [1, 2, 3, 4, 5],
  "base_tasks": ["salad_dressing", "ketchup"],   // [] = all base tasks
  "trials_per_variant": 50,
  "replan_steps": 8, "max_steps": 280,
  "num_workers": 12, "num_shards": 32,
  "save_videos": "none"
}
```

Perturbation keys: `light, background, noise, layout, camera, robot, language`.
`difficulties` are the benchmark's 1–5 labels (empirical; not a parameter magnitude).

## 4. Run + aggregate

```bash
source ../../.libero_plus/plus_env.sh          # MUJOCO_GL=egl, LIBERO_CONFIG_PATH, MAGICK_HOME
# start your policy server (step 2), then from the repo root:

# generate manifest + submit a slurm array (one task per shard; re-running resumes):
bash slurm/submit_plus_system.sh examples/libero/plus_system/configs/my_sweep.json
#   or just build the manifest:
"$LIBERO_PLUS_VENV/bin/python" examples/libero/plus_system/generate_manifest.py <config.json>

# aggregate when done -> all_episodes.csv/parquet + report:
python3 examples/libero/plus_system/aggregate_results.py my_sweep
```

Quick non-slurm smoke on one dimension:
```bash
"$LIBERO_PLUS_VENV/bin/python" main_plus.py --args.host localhost --args.port 8000 \
    --args.perturbation light --args.max-tasks 2
```

**Cost note:** sensor noise dominates compute (`motion_blur`/`zoom_blur`/`glass_blur` are
~10× slower per frame). Run cheap variants first with
`bash slurm/submit_plus_system.sh <cfg> --defer-noise-families motion_blur,zoom_blur`
(a runtime filter — does **not** change the manifest fingerprint; a later submit without
the flag collects the deferred variants). Do **not** put them in the config's
`exclude_noise_families` (that field is fingerprinted and strands collected episodes).

## Runtime environment

Clients need (from `plus_env.sh`): `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
`LIBERO_CONFIG_PATH=<venv>/.libero`, and `MAGICK_HOME` + `LD_LIBRARY_PATH` for `wand`.
Run clients on **gpu_a100** — MIG nodes have no EGL.

## Perturbation reference

Seven categories; exact mechanics in `LIBERO-plus/PERTURBATIONS.md`. In brief:
- **light**: rewrites the MuJoCo `<light>` (diffuse / direction / specular / castshadow grid).
- **layout**: displaces object placement regions — **basket included** — by up to `3·K` cm
  (uniform-in-disk, level `K`=1–5), or `_add_N` adds distractors. Object overlaps are
  avoided per-episode by rejection sampling at `env.reset()`, not by curated seeds.
- **camera**: procedural orbit (azimuth ±75°, elevation {0,15}°), dolly (1–2×), and ±10°
  aim jitter of **agentview** only; no FoV change.
- Also: background textures, sensor noise (blur/fog), robot init-state, language paraphrases.

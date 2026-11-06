#!/usr/bin/env bash
# build_pi_env.sh
#
# Rebuild pi_env — the pi0.5 training + serving environment — exactly as it was
# snapshotted, in the PARENT directory of this repo (…/openpi-fork/.. -> …/pi_env).
#
# It replays scripts/pi_env.requirements.txt (a full `uv pip freeze` of the live
# pi_env, 254 pinned packages incl. a git-pinned lerobot) into a fresh py3.11.15
# uv venv, then installs openpi + openpi-client editable from THIS repo. The
# editables go in with --no-deps so the pinned closure stays byte-for-byte.
#
# Usage:
#   scripts/build_pi_env.sh [TARGET_DIR]
#     TARGET_DIR   where to create the venv. Default: <repo>/../pi_env.
#
# Needs: uv on PATH, network (PyPI + the lerobot git SHA), ~8 GB disk.
# Idempotent guard: refuses to build over an existing TARGET_DIR.

set -euo pipefail

# uv pip commands honor an active VIRTUAL_ENV over --python in some paths; drop it
# so we only ever touch the venv we create here.
unset VIRTUAL_ENV

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$REPO_ROOT/scripts/pi_env.requirements.txt"
PY_VERSION="3.11.15"

TARGET="${1:-$(cd "$REPO_ROOT/.." && pwd)/pi_env}"

command -v uv >/dev/null || { echo "ERROR: need 'uv' on PATH (https://docs.astral.sh/uv/)" >&2; exit 1; }
[[ -f "$REQ" ]] || { echo "ERROR: missing $REQ" >&2; exit 1; }
if [[ -e "$TARGET" ]]; then
  echo "ERROR: $TARGET already exists. Remove it (rm -rf) or pass a different TARGET_DIR." >&2
  exit 1
fi

echo "== repo:    $REPO_ROOT"
echo "== target:  $TARGET  (python $PY_VERSION)"

uv venv --python "$PY_VERSION" "$TARGET"
PY="$TARGET/bin/python"

# 1) the exact frozen closure. --no-deps is REQUIRED, not an optimization: the live
#    pi_env was built incrementally and its installed set is not a valid global solve
#    (e.g. protobuf==7.35 violates google-api-core 2.24.2's declared protobuf<7). The
#    freeze already lists every transitive dep pinned, so we place them verbatim and
#    skip re-resolution. GIT_LFS_SKIP_SMUDGE keeps the lerobot git clone light.
echo "== installing frozen snapshot ($REQ)"
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$PY" --no-deps -r "$REQ"

# 2) the two editable workspace members, from this repo, without perturbing the closure
echo "== installing editable openpi + openpi-client (--no-deps)"
uv pip install --python "$PY" --no-deps -e "$REPO_ROOT" -e "$REPO_ROOT/packages/openpi-client"

echo "== sanity check"
"$PY" - <<'PY'
import openpi, jax, torch
import tensorflow as tf
print("openpi    :", openpi.__file__)
print("jax       :", jax.__version__)
print("torch     :", torch.__version__)
print("tensorflow:", tf.__version__)
PY

echo "== done -> $TARGET"

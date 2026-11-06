#!/usr/bin/env bash
# setup_libero_plus_env.sh
#
# Bootstraps the SIM side of the LIBERO-plus eval so the client in examples/libero/
# can run: clones LIBERO-plus, builds the py3.8 venv (libero_plus_env), installs
# deps, fetches the 6.4 GB assets, and writes the LIBERO config + known fixups.
#
# This sets up the ENVIRONMENT + CLIENT only. The policy SERVER (the model) is
# separate — the client talks to it over a websocket. For pi0.5 that server is
# openpi's scripts/serve_policy.py running in pi_env; a different model ships its
# own serve_*_websocket.py. See the banner printed at the end.
#
# Usage:
#   ./setup_libero_plus_env.sh [--prefix DIR] [--python PY] [--magick-home DIR]
#                              [--openpi-client SPEC] [--skip-assets] [--assets-only]
#
#   --prefix DIR       Where to create LIBERO-plus/ and libero_plus_env/.
#                      A RELATIVE path is resolved against THIS SCRIPT's dir
#                      (so it's reproducible wherever you run it from).
#                      Default: the script's own directory.
#   --python PY        Interpreter to build the venv (must be 3.8.x). Default: python3.8
#   --magick-home DIR  Prefix of an existing ImageMagick (has lib/ + bin/) for `wand`.
#                      Skips auto-install. On a no-root cluster, pass your conda/
#                      micromamba imagemagick env here.
#   --openpi-client S  pip spec/path for openpi-client (the client's only openpi dep).
#                      Default: auto-detect a sibling openpi repo's packages/openpi-client,
#                      else fall back to `pip install openpi-client`.
#   --skip-assets      Don't download assets.zip (reuse an existing unzip, or do it later).
#   --assets-only      Only (re)download+unpack assets into an existing clone, then exit.
#
# Idempotent: re-running skips steps whose outputs already exist.

set -euo pipefail

# ----------------------------- config / args --------------------------------
REPO_URL="https://github.com/sylvestf/LIBERO-plus.git"
REPO_REF="main"                                   # pin a commit/tag for reproducibility
HF_ASSETS_REPO="Sylvest/LIBERO-plus"              # dataset holding assets.zip
PYTHON_BIN="python3.8"
MAGICK_HOME_ARG=""
OPENPI_CLIENT=""
SKIP_ASSETS=0
ASSETS_ONLY=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$SCRIPT_DIR"                               # default; may be overridden

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)         PREFIX="$2"; shift 2;;
    --python)         PYTHON_BIN="$2"; shift 2;;
    --magick-home)    MAGICK_HOME_ARG="$2"; shift 2;;
    --openpi-client)  OPENPI_CLIENT="$2"; shift 2;;
    --skip-assets)    SKIP_ASSETS=1; shift;;
    --assets-only)    ASSETS_ONLY=1; shift;;
    -h|--help)        sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# Resolve a relative --prefix against the SCRIPT's dir (not $PWD).
case "$PREFIX" in
  /*) : ;;                                         # already absolute
  *)  PREFIX="$SCRIPT_DIR/$PREFIX";;
esac
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"

PLUS_DIR="$PREFIX/LIBERO-plus"
VENV="$PREFIX/libero_plus_env"
LIBERO_CFG_DIR="$VENV/.libero"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

echo "== prefix:        $PREFIX"
echo "== LIBERO-plus:   $PLUS_DIR"
echo "== venv:          $VENV"

# ----------------------------- helpers ---------------------------------------
die(){ echo "ERROR: $*" >&2; exit 1; }

download_assets() {
  [[ -d "$PLUS_DIR/libero/libero/assets/stable_scanned_objects" || \
     -d "$PLUS_DIR/libero/libero/assets/scenes/lights" ]] && \
     { echo "assets already present -> skip"; return 0; }
  echo "== downloading assets.zip (~6.4 GB) from HF $HF_ASSETS_REPO"
  "$PIP" install -q -U "huggingface_hub[cli]"
  local tmp; tmp="$(mktemp -d "$PREFIX/.assets_dl.XXXX")"
  "$VENV/bin/huggingface-cli" download "$HF_ASSETS_REPO" assets.zip \
      --repo-type dataset --local-dir "$tmp"
  echo "== unzipping"
  ( cd "$tmp" && "$PY" -c "import zipfile,sys; zipfile.ZipFile('assets.zip').extractall('unz')" )
  # The zip wraps the authors' server path; find the real assets/ dir inside.
  local src; src="$(find "$tmp/unz" -type d -name assets | head -1)"
  [[ -n "$src" ]] || die "no assets/ dir found inside assets.zip"
  mkdir -p "$PLUS_DIR/libero/libero"
  rsync -a "$src/" "$PLUS_DIR/libero/libero/assets/"
  rm -rf "$tmp"
  echo "== assets installed -> $PLUS_DIR/libero/libero/assets"
}

# ----------------------------- assets-only shortcut --------------------------
if [[ "$ASSETS_ONLY" == 1 ]]; then
  [[ -d "$PLUS_DIR" ]] || die "--assets-only needs an existing clone at $PLUS_DIR"
  [[ -x "$PY" ]] || die "--assets-only needs an existing venv at $VENV"
  download_assets
  exit 0
fi

# ----------------------------- 1. clone --------------------------------------
if [[ -d "$PLUS_DIR/.git" ]]; then
  echo "== clone exists -> skip"
else
  echo "== cloning $REPO_URL @ $REPO_REF"
  git clone "$REPO_URL" "$PLUS_DIR"
  git -C "$PLUS_DIR" checkout "$REPO_REF"
fi

# ----------------------------- 2. venv ---------------------------------------
if [[ -x "$PY" ]]; then
  echo "== venv exists -> skip create"
else
  command -v "$PYTHON_BIN" >/dev/null || die "need $PYTHON_BIN on PATH (pass --python). LIBERO-plus needs 3.8."
  "$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2]==(3,8), sys.version' \
     || die "$PYTHON_BIN is not 3.8.x"
  echo "== creating venv at $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$PIP" install -q -U pip wheel setuptools

# ----------------------------- 3. python deps --------------------------------
echo "== installing LIBERO-plus requirements + package"
"$PIP" install -r "$PLUS_DIR/requirements.txt"
"$PIP" install -e "$PLUS_DIR"                       # installs the `libero` package
"$PIP" install -r "$PLUS_DIR/extra_requirements.txt"   # wand, scikit-image
# small runtime deps the eval client needs on top of the above:
"$PIP" install imageio tyro tqdm

# openpi-client (the client's ONLY openpi dependency: image_tools + websocket client)
if [[ -z "$OPENPI_CLIENT" ]]; then
  for cand in "$SCRIPT_DIR"/../../packages/openpi-client \
              "$PREFIX"/openpi*/packages/openpi-client; do
    [[ -d "$cand" ]] && { OPENPI_CLIENT="$cand"; break; }
  done
fi
if [[ -n "$OPENPI_CLIENT" ]]; then
  echo "== installing openpi-client from $OPENPI_CLIENT"
  "$PIP" install -e "$OPENPI_CLIENT" 2>/dev/null || "$PIP" install "$OPENPI_CLIENT"
else
  echo "== openpi-client: trying PyPI (override with --openpi-client PATH if this fails)"
  "$PIP" install openpi-client || echo "WARN: openpi-client not installed; set --openpi-client"
fi

# ----------------------------- 4. ImageMagick (for wand) ---------------------
MAGICK_HOME=""
if [[ -n "$MAGICK_HOME_ARG" ]]; then
  MAGICK_HOME="$MAGICK_HOME_ARG"
elif "$PY" -c "import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library('MagickWand') else 1)" 2>/dev/null; then
  echo "== system MagickWand found -> no install needed"
elif command -v micromamba >/dev/null || command -v conda >/dev/null; then
  MM="$(command -v micromamba || command -v conda)"
  MAGICK_HOME="$PREFIX/imagemagick"
  if [[ ! -d "$MAGICK_HOME" ]]; then
    echo "== installing ImageMagick into $MAGICK_HOME via $(basename "$MM") (conda-forge)"
    "$MM" create -y -p "$MAGICK_HOME" -c conda-forge imagemagick
  fi
else
  echo "WARN: no ImageMagick found. Either run as root 'apt-get install libmagickwand-dev',"
  echo "      or re-run with --magick-home pointing at a conda/micromamba imagemagick env."
fi

# ----------------------------- 5. fixups -------------------------------------
# (a) belt-and-suspenders import path for the libero package
SITE="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
echo "$PLUS_DIR" > "$SITE/libero_plus_repo_path.pth"
# (b) silence robosuite's shared /tmp logfile (collides across users on login nodes)
printf 'FILE_LOGGING_LEVEL = None\n' > "$SITE/robosuite/macros_private.py" 2>/dev/null || \
  echo "WARN: robosuite not found yet; re-run after deps install if this warns"
# (c) LIBERO config pointing at THIS clone (client must export LIBERO_CONFIG_PATH=this dir)
mkdir -p "$LIBERO_CFG_DIR"
cat > "$LIBERO_CFG_DIR/config.yaml" <<YAML
benchmark_root: $PLUS_DIR/libero/libero
bddl_files: $PLUS_DIR/libero/libero/bddl_files
init_states: $PLUS_DIR/libero/libero/init_files
datasets: $PLUS_DIR/libero/datasets
assets: $PLUS_DIR/libero/libero/assets
YAML

# ----------------------------- 6. assets -------------------------------------
[[ "$SKIP_ASSETS" == 1 ]] && echo "== --skip-assets: not downloading assets" || download_assets

# ----------------------------- 7. env file the client sources ----------------
ENV_FILE="$PREFIX/plus_env.sh"
{
  echo "# source this before running the eval client"
  echo "export LIBERO_PLUS_VENV='$VENV'"
  echo "export LIBERO_CONFIG_PATH='$LIBERO_CFG_DIR'"
  echo "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl"
  if [[ -n "$MAGICK_HOME" ]]; then
    echo "export MAGICK_HOME='$MAGICK_HOME'"
    echo "export LD_LIBRARY_PATH='$MAGICK_HOME/lib':\${LD_LIBRARY_PATH:-}"
  fi
} > "$ENV_FILE"
echo "== wrote $ENV_FILE"

# ----------------------------- 8. smoke check --------------------------------
echo "== smoke check"
# shellcheck disable=SC1090
source "$ENV_FILE"
LIBERO_CONFIG_PATH="$LIBERO_CFG_DIR" "$PY" - <<'PY' || echo "SMOKE FAILED (check WARNs above)"
import libero, robosuite, scipy, cv2            # noqa
import wand.image                               # the hard ImageMagick dep
from libero.libero import get_libero_path
print("libero_plus_env OK:", get_libero_path("bddl_files"))
PY

cat <<BANNER

============================================================================
 SIM + CLIENT side is ready.  To RUN an eval you still need a POLICY SERVER:
   - pi0.5:  in openpi's pi_env:
       XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/serve_policy.py \\
         --port 8000 policy:checkpoint --policy.config <cfg> --policy.dir <ckpt>
   - other model: run that repo's serve_*_websocket.py (openpi websocket protocol).
 Then, in THIS env:
   source $ENV_FILE
   "$VENV/bin/python" <your examples/libero copy>/plus_system/run_shard.py \\
       --host localhost --port 8000 ...
============================================================================
BANNER

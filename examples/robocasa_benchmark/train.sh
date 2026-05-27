#!/usr/bin/env bash
#SBATCH --job-name=gr00t_robocasa_seen_pretrain
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# 2-GPU (B200) finetune of GR00T N1.7 on the robocasa-benchmark "seen tasks, pretrain split"
# soup, using scripts/gr00t_finetune_soup.py. Submit from the repo root:
#   sbatch examples/robocasa_benchmark/train.sh
# Override defaults via env, e.g.:
#   DATASET_SOUP=pretrain_atomic_seen MAX_STEPS=60000 GLOBAL_BATCH_SIZE=128 \
#     sbatch examples/robocasa_benchmark/train.sh
#
# Prereqs:
#   - .venv (uv) must be able to `import robocasa` (robosuite + your ../robocasa installed).
#   - RoboCasa dataset base path: set in robocasa/macros_private.py (DATASET_BASE_PATH),
#     or symlink data to the default robocasa/../datasets. pretrain_atomic_seen uses mg data.
set -euo pipefail

CYAN='\033[1;36m'; YELLOW='\033[1;33m'; GREEN='\033[1;32m'; RED='\033[1;31m'; NC='\033[0m'
log()  { echo -e "${CYAN}[train]${NC} $*"; }
warn() { echo -e "${YELLOW}[train]${NC} $*"; }
ok()   { echo -e "${GREEN}[train]${NC} $*"; }
fail() { echo -e "${RED}[train]${NC} $*" >&2; exit 1; }

# ── Resolve REPO_ROOT — SLURM copies the sbatch script to a spool dir, so BASH_SOURCE
#    and `git` won't find the repo. Priority: env > SLURM_SUBMIT_DIR > git > script grandparent.
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/gr00t_finetune_soup.py" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        _src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || _src_dir=""
        if [ -n "$_src_dir" ] && _gt="$(git -C "$_src_dir" rev-parse --show-toplevel 2>/dev/null)"; then
            REPO_ROOT="$_gt"
        elif [ -n "$_src_dir" ] && [ -f "$_src_dir/../../scripts/gr00t_finetune_soup.py" ]; then
            REPO_ROOT="$(cd "$_src_dir/../.." && pwd)"
        fi
    fi
fi
[ -n "${REPO_ROOT:-}" ] && [ -f "$REPO_ROOT/scripts/gr00t_finetune_soup.py" ] \
    || fail "cannot resolve REPO_ROOT. submit from repo root, or export REPO_ROOT=/path/to/Isaac-GR00T"
cd "$REPO_ROOT"
log "repo root: $REPO_ROOT"

# ── Run config (env-overridable)
DATASET_SOUP="${DATASET_SOUP:-pretrain_atomic_seen}"
MODALITY_CONFIG="examples/robocasa_benchmark/modality_config.py"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-ROBOCASA_PANDA_OMRON}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/robocasa_${DATASET_SOUP}}"
EXP_NAME="${EXP_NAME:-${DATASET_SOUP}_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-robocasa-gr00t-n1d7}"
USE_WANDB="${USE_WANDB:-1}"

NUM_GPUS=2                                    # matches #SBATCH --gres=gpu:2
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}" # per-GPU = 64; must be divisible by NUM_GPUS
MAX_STEPS="${MAX_STEPS:-60000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DS_WEIGHTS_ALPHA="${DS_WEIGHTS_ALPHA:-0.4}"
# Per-job master port to avoid clashes when two jobs land on one node.
MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"

(( GLOBAL_BATCH_SIZE % NUM_GPUS == 0 )) || fail "GLOBAL_BATCH_SIZE ($GLOBAL_BATCH_SIZE) must be divisible by NUM_GPUS ($NUM_GPUS)"

# ── Pre-flight checks (no silent fallbacks)
[ -d "$REPO_ROOT/.venv" ]        || fail "no .venv — set up the training env first"
[ -f "$MODALITY_CONFIG" ]        || fail "missing modality config: $MODALITY_CONFIG"

# ── ffmpeg runtime libs from conda env (torchcodec needs libavformat etc.)
if [ -z "${CONDA_FFMPEG_PREFIX:-}" ] && [ -f "$REPO_ROOT/examples/realworld/.ffmpeg_prefix" ]; then
    CONDA_FFMPEG_PREFIX="$(cat "$REPO_ROOT/examples/realworld/.ffmpeg_prefix")"
fi
[ -n "${CONDA_FFMPEG_PREFIX:-}" ] || fail "CONDA_FFMPEG_PREFIX not set — run setup_env.sh first or export it"
[ -d "$CONDA_FFMPEG_PREFIX/lib" ] || fail "no $CONDA_FFMPEG_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_FFMPEG_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_FFMPEG_PREFIX/bin:$PATH"
log "ffmpeg libs: $CONDA_FFMPEG_PREFIX/lib"

# activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# robocasa must import (soup resolution reads DATASET_SOUP_REGISTRY + each dataset's info.json)
python -c "from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY" 2>/dev/null \
    || fail "cannot import robocasa in .venv — install robosuite + your ../robocasa package"

if [ "$USE_WANDB" = 1 ]; then
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
        fail "USE_WANDB=1 but wandb not logged in. run: wandb login  (or set WANDB_API_KEY, or USE_WANDB=0)"
    fi
fi

mkdir -p "$OUTPUT_DIR" slurm_logs

# ── debug: run config
log "== run config =="
log "  job/node          = ${SLURM_JOB_ID:-NA} on ${SLURMD_NODENAME:-$(hostname)} (partition=${SLURM_JOB_PARTITION:-dgx-b200})"
log "  base_model        = $BASE_MODEL"
log "  dataset_soup      = $DATASET_SOUP"
log "  modality_config   = $MODALITY_CONFIG"
log "  embodiment_tag    = $EMBODIMENT_TAG"
log "  output_dir        = $OUTPUT_DIR"
log "  experiment_name   = $EXP_NAME"
log "  wandb             = $WANDB_PROJECT (USE_WANDB=$USE_WANDB)"
log "  num_gpus          = $NUM_GPUS   master_port=$MASTER_PORT"
log "  global_batch_size = $GLOBAL_BATCH_SIZE  (per-GPU = $((GLOBAL_BATCH_SIZE / NUM_GPUS)))"
log "  max_steps         = $MAX_STEPS  save_steps=$SAVE_STEPS  ds_weights_alpha=$DS_WEIGHTS_ALPHA"
log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# quick GPU sanity print
python - <<'PY'
import torch
from rich import print as rprint
rprint(f"[cyan][train][/cyan] torch={torch.__version__} cuda={torch.cuda.is_available()} n={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    rprint(f"[cyan][train][/cyan]   gpu[{i}] = {torch.cuda.get_device_name(i)}")
PY

ok "launching 2-GPU torchrun finetune"
exec torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    scripts/gr00t_finetune_soup.py \
    --dataset_soup        "$DATASET_SOUP" \
    --base_model_path     "$BASE_MODEL" \
    --embodiment_tag      "$EMBODIMENT_TAG" \
    --modality_config_path "$MODALITY_CONFIG" \
    --output_dir          "$OUTPUT_DIR" \
    --experiment_name     "$EXP_NAME" \
    --wandb_project       "$WANDB_PROJECT" \
    --num_gpus            "$NUM_GPUS" \
    --global_batch_size   "$GLOBAL_BATCH_SIZE" \
    --max_steps           "$MAX_STEPS" \
    --save_steps          "$SAVE_STEPS" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --ds_weights_alpha    "$DS_WEIGHTS_ALPHA" \
    $( [ "$USE_WANDB" = 1 ] && echo --use_wandb )

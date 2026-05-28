#!/usr/bin/env bash
#SBATCH --job-name=gr00t_robocasa_eval
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Eval a finetuned checkpoint: seen_tasks on the target split + unseen_tasks on the pretrain
# split. Starts the N1.7 ZMQ policy server, runs both client sweeps, then aggregates.
# The modality config is baked into the checkpoint's processor, so it is NOT passed here.
# Submit from the repo root (single gr00t_robocasa env that has both gr00t and robocasa):
#   CKPT=/path/to/checkpoints/robocasa_seen_pretrain/checkpoint-60000 sbatch examples/robocasa_benchmark/eval.sh
set -euo pipefail

CKPT="${CKPT:?set CKPT=/path/to/checkpoint-XXXXX}"
PORT="${PORT:-5555}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-5}"

# start the policy server in the background; kill it on exit
python gr00t/eval/run_gr00t_server.py \
    --model-path "$CKPT" \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper \
    --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# wait until the server has loaded the model and answers ping (up to ~30 min)
for _ in $(seq 1 120); do
    if python -c "import sys; from gr00t.policy.server_client import PolicyClient; sys.exit(0 if PolicyClient(host='127.0.0.1', port=$PORT).ping() else 1)" 2>/dev/null; then
        echo "[eval] server ready on port $PORT"; break
    fi
    sleep 15
done

# seen tasks -> target split
python scripts/run_robocasa_eval.py \
    --output_dir "$CKPT" --task_set seen_tasks --split target \
    --policy_client_host 127.0.0.1 --policy_client_port "$PORT" \
    --n_episodes "$N_EPISODES" --n_envs "$N_ENVS"

# unseen tasks -> pretrain split
python scripts/run_robocasa_eval.py \
    --output_dir "$CKPT" --task_set unseen_tasks --split pretrain \
    --policy_client_host 127.0.0.1 --policy_client_port "$PORT" \
    --n_episodes "$N_EPISODES" --n_envs "$N_ENVS"

# aggregate per-task stats.json into the seen/unseen groups
python gr00t/eval/get_eval_stats.py --dir "$CKPT" --task_set seen_tasks unseen_tasks

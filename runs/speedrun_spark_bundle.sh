#!/bin/bash

# This script trains your own GPT-2 grade LLM on a DGX Spark Bundle (2x GB10).
# It is the Spark Bundle equivalent of runs/speedrun.sh (designed for 8xH100).
# Full pipeline: dataset download --> tokenizer --> pretrain --> eval --> SFT --> chat eval.
# Expected time: ~3.1 days total (~2.6 days pretraining). Measured on DGX OS 7.4.0.
#
# Optimizations enabled automatically:
#   - Flash Attention 2 (native sliding window, 28% faster than SDPA)
#   - SSSL window pattern (Karpathy's default, better model quality)
#   - FP8 training (12% faster matmuls, less memory)
#   - RDMA/RoCE over stacking cable (200Gbps, near-perfect 2x scaling)
#   - torch.compile with persistent cache (3.2x faster restarts)
#
# Requirements:
#   - 2x DGX Spark connected via QSFP/CX7 stacking cable
#   - NGC container: docker pull nvcr.io/nvidia/pytorch:26.03-py3
#   - Stacking cable configured via NVIDIA discover-sparks playbook
#
# =============================================================================
# Setup: start NGC containers on BOTH Sparks
# =============================================================================
#
# Open two terminal tabs on your MacBook (or wherever you SSH from).
#
# --- Terminal 1: spark-403d ---
#   ssh spark-403d.local
#   docker run --gpus all -it --rm \
#       --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 \
#       --cap-add SYS_RESOURCE \
#       --device /dev/infiniband \
#       -v ~/nanochat:/workspace/nanochat \
#       -v ~/.cache/nanochat:/root/.cache/nanochat \
#       -v ~/.cache/torchinductor:/tmp/torchinductor_root \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       -v ~/.ssh:/root/.ssh:ro \
#       nvcr.io/nvidia/pytorch:26.03-py3 bash
#   cd /workspace/nanochat && pip install --no-deps -e . && \
#       pip install fastapi kernels rustbpe tiktoken uvicorn wandb
#
# --- Terminal 2: spark-392c ---
#   (same docker run command, same pip install)
#
# NOTE: --ulimit flags are recommended by NVIDIA in the NGC container startup
#   banner (printed every time the container starts).
# NOTE: --cap-add SYS_RESOURCE allows HumanEval's code sandbox to set resource
#   limits (setrlimit) beyond Docker's default ulimits. Without it, HumanEval
#   evaluation scores 0%. Karpathy doesn't hit this because speedrun.sh runs
#   on bare metal (no Docker). See nanochat/execution.py:152.
# NOTE: -v ~/.ssh:/root/.ssh:ro mounts SSH keys (read-only) so the script can
#   sync checkpoints between Sparks over the stacking cable (via scp).
# NOTE: -v ~/.cache/huggingface persists HF token and downloaded datasets across
#   container restarts (saves ~5 min of SFT data re-download). Token is protected
#   by chmod 600 on the host. Store it securely with:
#   read -sp "Token: " T && mkdir -p ~/.cache/huggingface &&
#   echo "$T" > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
#   Never pass tokens via -e HF_TOKEN=... (visible in docker inspect and ps).
#
# =============================================================================
# Usage
# =============================================================================
#
# 1) Full pipeline (default -- like Karpathy's speedrun.sh):
#      NODE_RANK=0 bash runs/speedrun_spark_bundle.sh   # on spark-403d
#      NODE_RANK=1 bash runs/speedrun_spark_bundle.sh   # on spark-392c
#
# 2) Run a single stage:
#      STAGE=pretrain NODE_RANK=0 bash runs/speedrun_spark_bundle.sh
#      STAGE=sft      NODE_RANK=0 bash runs/speedrun_spark_bundle.sh
#
#    Available stages: dataset, tokenizer, pretrain, eval, sft, chateval
#    Stages dataset/tokenizer/eval/chateval run on rank 0 only.
#    Stages pretrain/sft require both ranks.
#
# 3) Resume pretraining after a crash:
#      RESUME_STEP=500 NODE_RANK=0 bash runs/speedrun_spark_bundle.sh
#
# 4) Enable wandb logging:
#      WANDB_RUN=my_experiment NODE_RANK=0 bash runs/speedrun_spark_bundle.sh
#
# 5) Quick experiment at small scale (~5 min):
#      DEPTH=12 NODE_RANK=0 bash runs/speedrun_spark_bundle.sh
#
# =============================================================================

set -e

# --- Configuration (edit these for your setup) ---
DEPTH="${DEPTH:-24}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"  # 16 stays below the 110 GB host-memory budget
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
NVFP4="${NVFP4:-1}"          # native GB10 NVFP4 training; set 0 to fall back to FP8
PRECISION_FLAG="--fp8"
[ "$NVFP4" = "1" ] && PRECISION_FLAG="--nvfp4"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
DATASET_SHARDS="${DATASET_SHARDS:-170}"
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
SAVE_EVERY="${SAVE_EVERY:-250}"
EVAL_EVERY="${EVAL_EVERY:-250}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"

# --- Bundle networking ---
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
RANK1_ADDR="${RANK1_ADDR:-192.168.100.11}"
MASTER_PORT="${MASTER_PORT:-29500}"
SYNC_PORT="${SYNC_PORT:-29501}"     # TCPStore port for between-stage sync (avoids torchrun's port)
NCCL_IFNAME="${NCCL_IFNAME:-enp1s0f1np1}"

# --- Environment ---
NODE_RANK="${NODE_RANK:?ERROR: Set NODE_RANK=0 on spark-403d or NODE_RANK=1 on spark-392c}"
WANDB_RUN="${WANDB_RUN:-dummy}"
RESUME_STEP="${RESUME_STEP:-}"
STAGE="${STAGE:-all}"
export OMP_NUM_THREADS=1
export NANOCHAT_COMPILE_THREADS="${NANOCHAT_COMPILE_THREADS:-1}"
# Keep native/Triton build helpers single threaded as well.  Inductor's Python
# worker limit above is the main guard; this avoids a second compiler fan-out
# from exhausting GB10's unified DRAM during the first NVFP4 build.
export MAX_JOBS="${MAX_JOBS:-1}"
export NCCL_SOCKET_IFNAME="$NCCL_IFNAME"
export UCX_NET_DEVICES="$NCCL_IFNAME"
# NVIDIA playbook-recommended NCCL settings for stacked Sparks.
# NCCL_IB_HCA lists all RDMA devices (both logical interfaces per CX-7 port).
# NCCL_IB_SUBNET_AWARE_ROUTING helps NCCL pick the right interface for each peer.
# Ref: github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/nccl
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1}"
export NCCL_IB_SUBNET_AWARE_ROUTING="${NCCL_IB_SUBNET_AWARE_ROUTING:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
# The Spark container is recreated after a reboot.  Keep the small tokenizer
# wheels in the persistent nanochat cache so a fresh container can import them
# without reinstalling into the image, and prepend that site directory when it
# exists on both ranks.
NANOCHAT_SITE_DIR="${NANOCHAT_SITE_DIR:-$NANOCHAT_BASE_DIR/site}"
if [ -d "$NANOCHAT_SITE_DIR" ]; then
    export PYTHONPATH="$NANOCHAT_SITE_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

# --- Helper: should we run this stage? ---
should_run() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# --- Helper: find latest valid checkpoint step (empty string if none) ---
# Uses runs/checkpoint_utils.py to validate checkpoints and handle corruption.
# See tests/test_checkpoint_validation.py for unit tests.
find_latest_checkpoint() {
    local dir="$1"
    python3 runs/checkpoint_utils.py "$dir" 2>/dev/null
}

# --- Helper: run distributed training ---
run_distributed() {
    local resume_args=""
    if [ -n "$RESUME_STEP" ]; then
        resume_args="--resume-from-step=$RESUME_STEP"
        echo "Resuming from step $RESUME_STEP"
    fi
    torchrun \
        --nproc_per_node=1 --nnodes=2 --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        -m "$@" $resume_args
}

echo "=============================================="
echo "DGX Spark Bundle Speedrun"
echo "=============================================="
echo "Node rank:      $NODE_RANK"
echo "Depth:          $DEPTH"
echo "Batch size:     $DEVICE_BATCH_SIZE"
echo "Window pattern: $WINDOW_PATTERN"
echo "Precision:      $([ "$NVFP4" = "1" ] && echo NVFP4 || echo FP8)"
echo "Compile threads: $NANOCHAT_COMPILE_THREADS"
echo "Compiler jobs:  $MAX_JOBS"
echo "Stage:          $STAGE"
echo "Wandb:          $WANDB_RUN"
[ -n "$RESUME_STEP" ] && echo "Resume from:    step $RESUME_STEP"
echo "=============================================="

# =============================================================================
# Preflight checks
# =============================================================================
# Verify stacking cable SSH works. NVIDIA's discover-sparks playbook sets up
# passwordless SSH between the Sparks using a shared key (~/.ssh/id_ed25519_shared).
# This is required for checkpoint syncing between pretrain and SFT stages.
# The Docker container must mount SSH keys: -v ~/.ssh:/root/.ssh:ro
#
# NOTE: We specify -i for the key explicitly because the host's ~/.ssh/config
# is owned by the host user, not root -- SSH inside the container rejects it
# ("Bad owner or permissions on /root/.ssh/config").
# SSH options for checkpoint syncing inside the container.
# -F /dev/null: skip ~/.ssh/config (owned by host user, rejected by root in container)
# -i: use the NVIDIA discover-sparks shared key explicitly
REMOTE_USER="${REMOTE_USER:-lycis}"
SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519_shared}"
SSH_OPTS="-F /dev/null -i $SSH_KEY -o ConnectTimeout=5 -o StrictHostKeyChecking=no"

if [ "$NODE_RANK" = "0" ]; then
    REMOTE_ADDR="$RANK1_ADDR"
else
    REMOTE_ADDR="$MASTER_ADDR"
fi

if ! ssh $SSH_OPTS "$REMOTE_USER@$REMOTE_ADDR" hostname >/dev/null 2>&1; then
    echo "ERROR: Cannot SSH to $REMOTE_USER@$REMOTE_ADDR from inside the container"
    echo ""
    echo "Checkpoint syncing requires SSH between Sparks over the stacking cable."
    echo "1. Run NVIDIA's discover-sparks playbook: https://build.nvidia.com/spark/connect-two-sparks"
    echo "2. Verify on the host: ssh $REMOTE_ADDR hostname"
    echo "3. Mount SSH keys in Docker: -v ~/.ssh:/root/.ssh:ro"
    echo "4. If your username is not 'matt', set REMOTE_USER=yourname"
    echo "5. If the key is not id_ed25519_shared, set SSH_KEY=/root/.ssh/your_key"
    exit 1
fi
echo "Preflight: SSH to $REMOTE_USER@$REMOTE_ADDR OK"

# =============================================================================
# Stage 1: Dataset download (BOTH ranks -- each rank reads its own shards)
# =============================================================================
# The distributed dataloader shards data across ranks: each rank reads different
# row groups from the same parquet files. Both Sparks must have a local copy.
# The download is idempotent -- existing files are skipped.

if should_run "dataset"; then
    echo ""
    echo "--- Stage: dataset ---"
    python -m nanochat.dataset -n "$DATASET_SHARDS"
fi

# =============================================================================
# Stage 2: Tokenizer (BOTH ranks -- each rank tokenizes its own data)
# =============================================================================
# The tokenizer is trained deterministically on the same data, so both ranks
# produce identical results. Each rank needs a local copy for the dataloader.

if should_run "tokenizer"; then
    echo ""
    echo "--- Stage: tokenizer ---"
    python -m scripts.tok_train
    python -m scripts.tok_eval
fi

# =============================================================================
# Stage 3: Pretraining (both ranks, distributed)
# =============================================================================
# This is the main event: ~5.1 days for depth=24.
# CORE metric is disabled during pretrain (--core-metric-every=-1) because the
# CORE eval takes ~10 min and causes NCCL timeout in multi-node setups. The eval
# stage runs CORE separately after pretraining completes.
# Checkpoints every --save-every steps. If interrupted, just re-run -- the script
# auto-detects the latest checkpoint and resumes. Or set RESUME_STEP=N explicitly.

if should_run "pretrain"; then
    echo ""
    echo "--- Stage: pretrain ---"
    # Auto-detect latest checkpoint for resume (if RESUME_STEP not set explicitly)
    if [ -z "$RESUME_STEP" ]; then
        CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d${DEPTH}"
        LATEST=$(find_latest_checkpoint "$CHECKPOINT_DIR")
        if [ -n "$LATEST" ]; then
            RESUME_STEP="$LATEST"
            echo "Auto-detected checkpoint at step $RESUME_STEP -- resuming"
        fi
    fi
    run_distributed scripts.base_train -- \
        --depth="$DEPTH" \
        --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --window-pattern="$WINDOW_PATTERN" \
        $PRECISION_FLAG \
        --run="$WANDB_RUN" \
        --save-every="$SAVE_EVERY" \
        --eval-every="$EVAL_EVERY" \
        --core-metric-every="$CORE_METRIC_EVERY" \
        --sample-every="$SAMPLE_EVERY" \
        --num-iterations="$NUM_ITERATIONS"
fi

# =============================================================================
# Checkpoint sync + barrier: deliver pretrained model to rank 1
# =============================================================================
# Pretraining saves the model checkpoint on rank 0 only (standard PyTorch DDP).
# SFT needs the checkpoint on both nodes. Rank 0 scp's it over the stacking
# cable, then both ranks synchronize via a PyTorch TCPStore barrier.
#
# The barrier uses PyTorch's TCPStore -- the same mechanism torchrun uses
# internally for node rendezvous. No polling, no filesystem watching, no
# timing assumptions. See runs/sync_barrier.py for details.

if should_run "sft"; then
    # Rank 1: make checkpoint dir writable for incoming scp.
    # Docker runs as root, so files created during pretrain are root-owned.
    # The scp from rank 0 connects as $REMOTE_USER (not root), so it needs
    # write permission on the directory.
    if [ "$NODE_RANK" = "1" ]; then
        CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d${DEPTH}"
        mkdir -p "$CHECKPOINT_DIR"
        chmod 777 "$CHECKPOINT_DIR"
    fi

    # Rank 0: scp checkpoint to rank 1
    # NOTE: Inside the container, NANOCHAT_BASE_DIR is /root/.cache/nanochat.
    # But scp connects to rank 1's HOST (as $REMOTE_USER), where the path is
    # ~/.cache/nanochat. We use ~ so SSH resolves it to the remote user's home.
    REMOTE_CHECKPOINT_DIR="~/.cache/nanochat/base_checkpoints/d${DEPTH}"
    if [ "$NODE_RANK" = "0" ]; then
        CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d${DEPTH}"
        SYNC_STEP=$(find_latest_checkpoint "$CHECKPOINT_DIR")
        if [ -n "$SYNC_STEP" ]; then
            STEP_PAD=$(printf "%06d" "$SYNC_STEP")
            echo "Syncing pretrained model (step $SYNC_STEP) to rank 1..."
            ssh $SSH_OPTS "$REMOTE_USER@$RANK1_ADDR" "mkdir -p $REMOTE_CHECKPOINT_DIR"
            scp $SSH_OPTS \
                "$CHECKPOINT_DIR/model_${STEP_PAD}.pt" \
                "$CHECKPOINT_DIR/meta_${STEP_PAD}.json" \
                "$REMOTE_USER@$RANK1_ADDR:$REMOTE_CHECKPOINT_DIR/"
        fi
    fi

    # Barrier: both ranks block until both arrive (TCPStore, no polling)
    echo "Barrier: waiting for both ranks to be ready for SFT..."
    python3 runs/sync_barrier.py \
        --rank "$NODE_RANK" --addr "$MASTER_ADDR" \
        --key checkpoint_synced --port "$SYNC_PORT"
    echo "Barrier: both ranks ready"
fi

# =============================================================================
# Stage 4: SFT -- Supervised Fine-Tuning (both ranks, distributed)
# =============================================================================
# Teaches the model conversation format, tool use, and multiple choice.
# Downloads synthetic identity conversations on first run.
#
# NOTE: SFT runs BEFORE eval so both distributed stages (pretrain, SFT) are
# back-to-back. This prevents a race condition where rank 1 starts torchrun
# for SFT while rank 0 is still running single-GPU eval.

if should_run "sft"; then
    echo ""
    echo "--- Stage: sft ---"
    # Download identity conversations (both ranks need it, tiny file)
    if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
        curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
            https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    fi
    # Clear RESUME_STEP -- SFT doesn't support resume, always runs fresh.
    # Disable inline evals (--chatcore-every=-1 --eval-every=-1) for the same
    # reason as pretrain: ChatCORE generates text on rank 0 only, which takes
    # 10+ min and causes NCCL timeout on rank 1. The chateval stage runs
    # ChatCORE separately after SFT, outside of torchrun.
    RESUME_STEP="" run_distributed scripts.chat_sft -- \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --run="$WANDB_RUN" \
        --chatcore-every=-1 \
        --eval-every=-1
fi

# =============================================================================
# Stage 5: Base model evaluation (rank 0 only)
# =============================================================================

if should_run "eval" && [ "$NODE_RANK" = "0" ]; then
    echo ""
    echo "--- Stage: eval ---"
    python -m scripts.base_eval --device-batch-size="$DEVICE_BATCH_SIZE"
fi

# =============================================================================
# Stage 6: Chat evaluation (rank 0 only)
# =============================================================================

if should_run "chateval" && [ "$NODE_RANK" = "0" ]; then
    echo ""
    echo "--- Stage: chateval ---"
    python -m scripts.chat_eval -i sft
fi

# =============================================================================
# Done
# =============================================================================

echo ""
echo "=============================================="
echo "Speedrun complete!"
echo "=============================================="
if [ "$NODE_RANK" = "0" ]; then
    echo ""
    echo "Chat with your model:"
    echo "  python -m scripts.chat_cli -p 'Why is the sky blue?'"
    echo ""
    echo "Or start the web UI:"
    echo "  python -m scripts.chat_web"
fi

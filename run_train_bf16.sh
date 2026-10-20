#!/bin/bash
# Two-node BF16 (BF16 master) throughput check.
set -e
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1}"
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NANOCHAT_COMPILE_THREADS="${NANOCHAT_COMPILE_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-1}"
export OMP_NUM_THREADS=1
export NANOCHAT_MASTER_BF16=1
cd /workspace/nanochat
exec torchrun --nproc_per_node=1 --nnodes=2 --node_rank="$1" \
  --master_addr="${MASTER_ADDR:-192.168.100.10}" --master_port="${MASTER_PORT:-29590}" \
  -m scripts.base_train -- \
  --depth=24 --device-batch-size=16 --window-pattern=SSSL \
  --run=dummy --model-tag=bf16bench \
  --save-every=100000 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
  --num-iterations="${STEPS:-4}"

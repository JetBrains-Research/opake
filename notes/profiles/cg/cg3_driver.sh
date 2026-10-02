#!/bin/bash
# DPTrainer with the CUDA-graphed chunk (after the capturable-mask fix), vs eager in the same session.
set -u
cd ~/opaque; mkdir -p runs/cg
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
DT="--model-name Qwen/Qwen2.5-Coder-7B --dataset JetBrains/KStack --dataset-text-field content --dtype bfloat16 --lora-r 16 --lora-alpha 16 --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-eval-on-start --use-performance-kernels --target-epsilon 3.0 --no-wandb --log-steps 1"
run() { local P=$1 ENVS=$2; shift 2
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/cg/$P.smi & local SMI=$!
  env $ENVS timeout 1500 $PY $W "$@" > runs/cg/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run dt_cg_mb2    "CB_CUDAGRAPH=1"              examples/train_dpsgd_trainer.py $DT --clip-backend torch  --microbatch-size 2 --output-dir /tmp/dt_cg_mb2
run dt_mb4       "CB_X=0"                      examples/train_dpsgd_trainer.py $DT --clip-backend triton --microbatch-size 4 --output-dir /tmp/dt_mb4
run dt_cg_mb4    "CB_CUDAGRAPH=1"              examples/train_dpsgd_trainer.py $DT --clip-backend torch  --microbatch-size 4 --output-dir /tmp/dt_cg_mb4
run dt_np_mb4    "CB_NO_PEFT=1"                examples/train_dpsgd_trainer.py $DT --clip-backend triton --microbatch-size 4 --output-dir /tmp/dt_np_mb4
run dt_cg_np_mb4 "CB_CUDAGRAPH=1 CB_NO_PEFT=1" examples/train_dpsgd_trainer.py $DT --clip-backend torch  --microbatch-size 4 --output-dir /tmp/dt_cg_np_mb4
echo CG3DONE

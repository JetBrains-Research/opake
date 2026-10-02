#!/bin/bash
# Kernel-level profiles of the current best configurations (kernel-targets report).
set -u
cd ~/opaque; mkdir -p runs/kp
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --clip-backend triton --microbatch-size 2"
run() { local P=$1; shift
  echo "=== $P start $(date -u +%T)"
  CB_KPROF=runs/kp/$P CB_KPROF_STEPS="4:cuda,5:all" $PY $W "$@" > runs/kp/$P.log 2>&1
  echo "=== $P exit $? $(date -u +%T)"
}
run kp_eager examples/train_dpsgd.py $TD --no-kernel-patches
run kp_kern examples/train_dpsgd.py $TD
echo KPDONE

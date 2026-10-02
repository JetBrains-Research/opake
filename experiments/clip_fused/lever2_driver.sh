#!/bin/bash
# Clean reruns + gradient checkpointing x larger microbatches (kernel-targets report).
set -u
cd ~/opaque; mkdir -p runs/lv
until grep -q LVDONE ~/lever_driver.log 2>/dev/null; do sleep 20; done
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-kernel-patches --clip-backend triton"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1; shift
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/lv/$P.smi & local SMI=$!
  timeout 1500 $PY $W "$@" --memory-profile runs/lv/$P $MP > runs/lv/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run lv_base2   examples/train_dpsgd.py $TD --microbatch-size 2
run lv_mb4b    examples/train_dpsgd.py $TD --microbatch-size 4
run lv_gc_mb4  examples/train_dpsgd.py $TD --microbatch-size 4 --gradient-checkpointing
run lv_gc_mb8  examples/train_dpsgd.py $TD --microbatch-size 8 --gradient-checkpointing
run lv_gc_mb16 examples/train_dpsgd.py $TD --microbatch-size 16 --gradient-checkpointing
echo LV2DONE

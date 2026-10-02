#!/bin/bash
# torch.compile without Opake's PEFT LoRA kernels + best eager combinations (kernel-targets report).
set -u
cd ~/opaque; mkdir -p runs/lv
until grep -q LV2DONE ~/lever2_driver.log 2>/dev/null; do sleep 20; done
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-kernel-patches"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1 ENVS=$2; shift 2
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/lv/$P.smi & local SMI=$!
  env $ENVS timeout 1800 $PY $W "$@" --memory-profile runs/lv/$P $MP > runs/lv/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run lv_np_mb4    "CB_NO_PEFT=1" examples/train_dpsgd.py $TD --clip-backend triton --microbatch-size 4
run lv_np_gc_mb8 "CB_NO_PEFT=1" examples/train_dpsgd.py $TD --clip-backend triton --microbatch-size 8 --gradient-checkpointing
run lv_np_comp   "CB_NO_PEFT=1" examples/train_dpsgd.py $TD --clip-backend auto --microbatch-size 2 --torch-compile
run lv_np_compro "CB_NO_PEFT=1" examples/train_dpsgd.py $TD --clip-backend auto --microbatch-size 2 --torch-compile --torch-compile-mode reduce-overhead
echo LV3DONE

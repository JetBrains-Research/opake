#!/bin/bash
# GPU-side levers once host overhead is amortized (mb=4) + kernel profile of the best eager config.
set -u
cd ~/opaque; mkdir -p runs/kp runs/lv
until grep -q LV4DONE ~/lever4_driver.log 2>/dev/null; do sleep 20; done
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-kernel-patches --clip-backend triton --microbatch-size 4"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1 ENVS=$2; shift 2
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/lv/$P.smi & local SMI=$!
  env $ENVS timeout 1500 $PY $W "$@" > runs/lv/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run lv_tf32_mb4    "CB_TF32=1"              examples/train_dpsgd.py $TD --memory-profile runs/lv/lv_tf32_mb4 $MP
run lv_np_tf32_mb4 "CB_NO_PEFT=1 CB_TF32=1" examples/train_dpsgd.py $TD --memory-profile runs/lv/lv_np_tf32_mb4 $MP
echo "=== kp_np_mb4 start $(date -u +%T)"
CB_NO_PEFT=1 CB_KPROF=runs/kp/kp_np_mb4 CB_KPROF_STEPS="4:cuda" timeout 1500 $PY $W examples/train_dpsgd.py $TD > runs/kp/kp_np_mb4.log 2>&1
echo "=== kp_np_mb4 exit $? $(date -u +%T)"
echo LV5DONE

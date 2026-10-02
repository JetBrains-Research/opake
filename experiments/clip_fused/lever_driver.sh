#!/bin/bash
# Cheap host-overhead and precision levers on the 7B workload (kernel-targets report).
set -u
cd ~/opaque; mkdir -p runs/lv
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1 ENVS=$2; shift 2
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/lv/$P.smi & local SMI=$!
  env $ENVS timeout 1500 $PY $W "$@" --memory-profile runs/lv/$P $MP > runs/lv/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run lv_base   "CB_X=0"       examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend triton --microbatch-size 2
run lv_mb4    "CB_X=0"       examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend triton --microbatch-size 4
run lv_tf32   "CB_TF32=1"    examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend triton --microbatch-size 2
run lv_nopeft "CB_NO_PEFT=1" examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend triton --microbatch-size 2
run lv_comp   "CB_X=0"       examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend auto --microbatch-size 2 --torch-compile
run lv_compro "CB_X=0"       examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend auto --microbatch-size 2 --torch-compile --torch-compile-mode reduce-overhead
run lv_kcomp  "CB_X=0"       examples/train_dpsgd.py $TD --clip-backend auto --microbatch-size 2 --torch-compile
echo LVDONE

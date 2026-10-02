#!/bin/bash
# CUDA-graphed chunk kernel on the 7B workload: correctness check + timing (same session, identical batches).
set -u
cd ~/opaque; mkdir -p runs/cg
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1 ENVS=$2; shift 2
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/cg/$P.smi & local SMI=$!
  env $ENVS timeout 1500 $PY $W "$@" > runs/cg/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run cg_check3   "CB_CUDAGRAPH=1 CB_CUDAGRAPH_CHECK=1" examples/train_dpsgd.py ${TD/--stop-at-step 6/--stop-at-step 4} --no-kernel-patches --clip-backend torch --microbatch-size 2
run base_mb2    "CB_X=0"           examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend triton --microbatch-size 2 --memory-profile runs/cg/base_mb2 $MP
run cg_mb2      "CB_CUDAGRAPH=1"   examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend torch --microbatch-size 2 --memory-profile runs/cg/cg_mb2 $MP
run cg_kern_mb2 "CB_CUDAGRAPH=1"   examples/train_dpsgd.py $TD --clip-backend torch --microbatch-size 2 --memory-profile runs/cg/cg_kern_mb2 $MP
run cg_mb4      "CB_CUDAGRAPH=1"   examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend torch --microbatch-size 4 --memory-profile runs/cg/cg_mb4 $MP
run base_best   "CB_NO_PEFT=1 CB_TF32=1"                examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend triton --microbatch-size 4 --memory-profile runs/cg/base_best $MP
run cg_best     "CB_CUDAGRAPH=1 CB_NO_PEFT=1 CB_TF32=1" examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend torch --microbatch-size 4 --memory-profile runs/cg/cg_best $MP
run cg_kern_tf32_mb4 "CB_CUDAGRAPH=1 CB_TF32=1"         examples/train_dpsgd.py $TD --clip-backend torch --microbatch-size 4 --memory-profile runs/cg/cg_kern_tf32_mb4 $MP
echo "=== kp_cg_mb2 start $(date -u +%T)"
CB_CUDAGRAPH=1 CB_KPROF=runs/cg/kp_cg_mb2 CB_KPROF_STEPS="4:cuda" timeout 1500 $PY $W examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend torch --microbatch-size 2 > runs/cg/kp_cg_mb2.log 2>&1
echo "=== kp_cg_mb2 exit $? $(date -u +%T)"
echo CGDONE

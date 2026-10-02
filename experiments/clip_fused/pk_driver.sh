#!/bin/bash
# Packaged multi-tensor clip vs the previous per-tensor kernels (report §12.3).
set -u
cd ~/opaque; mkdir -p runs
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --clip-backend triton --microbatch-size 2"
DT="--model-name Qwen/Qwen2.5-Coder-7B --dataset JetBrains/KStack --dataset-text-field content --dtype bfloat16 --lora-r 16 --lora-alpha 16 --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-eval-on-start --use-performance-kernels --target-epsilon 3.0 --no-wandb --log-steps 1 --clip-backend triton --microbatch-size 2"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1 OLD=$2 NOFE=$3; shift 3
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/$P.smi & local SMI=$!
  CB_OLD=$OLD CB_NOFOREACH=$NOFE $PY $W "$@" > runs/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run pk_eager_old 1 0 examples/train_dpsgd.py $TD --no-kernel-patches --memory-profile runs/pk_eager_old $MP
run pk_eager_new 0 0 examples/train_dpsgd.py $TD --no-kernel-patches --memory-profile runs/pk_eager_new $MP
run pk_eager_nofe 0 1 examples/train_dpsgd.py $TD --no-kernel-patches --memory-profile runs/pk_eager_nofe $MP
run pk_kern_old 1 0 examples/train_dpsgd.py $TD --memory-profile runs/pk_kern_old $MP
run pk_kern_new 0 0 examples/train_dpsgd.py $TD --memory-profile runs/pk_kern_new $MP
CL="${TD/--stop-at-step 6/--stop-at-step 8} --no-kernel-patches"
run pk_clean_old 1 0 examples/train_dpsgd.py $CL
run pk_clean_new 0 0 examples/train_dpsgd.py $CL
run pk_dt_old 1 0 examples/train_dpsgd_trainer.py $DT --output-dir /tmp/pk_dt_old
run pk_dt_new 0 0 examples/train_dpsgd_trainer.py $DT --output-dir /tmp/pk_dt_new
echo PKDONE

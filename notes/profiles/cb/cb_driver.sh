#!/bin/bash
set -u
cd ~/opaque; mkdir -p runs
export WANDB_MODE=disabled
PY=.venv/bin/python
TD="--preset qwen-7b-kstack --max-seq-len 512 --microbatch-size 2 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100"
DT="--model-name Qwen/Qwen2.5-Coder-7B --dataset JetBrains/KStack --dataset-text-field content --dtype bfloat16 --lora-r 16 --lora-alpha 16 --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --max-seq-len 512 --batch-size 16 --microbatch-size 2 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --no-eval-on-start --use-performance-kernels --target-epsilon 3.0 --no-wandb --log-steps 1"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1; shift
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/$P.smi & local SMI=$!
  "$@" > runs/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run cb_dt_torch       $PY examples/train_dpsgd_trainer.py $DT --output-dir /tmp/cb_dt_torch
run cb_dt_triton      $PY examples/train_dpsgd_trainer.py $DT --output-dir /tmp/cb_dt_triton --clip-backend triton
run cb_dt_save        $PY examples/train_dpsgd_trainer.py $DT --output-dir /tmp/cb_dt_save --no-lora-mlp-recompute
run cb_dt_triton_save $PY examples/train_dpsgd_trainer.py $DT --output-dir /tmp/cb_dt_triton_save --clip-backend triton --no-lora-mlp-recompute
run cb_kern_torch     $PY examples/train_dpsgd.py $TD --memory-profile runs/cb_kern_torch $MP
run cb_kern_triton    $PY examples/train_dpsgd.py $TD --memory-profile runs/cb_kern_triton $MP --clip-backend triton
run cb_eager_torch    $PY examples/train_dpsgd.py $TD --memory-profile runs/cb_eager_torch $MP --no-kernel-patches
run cb_eager_triton   $PY examples/train_dpsgd.py $TD --memory-profile runs/cb_eager_triton $MP --no-kernel-patches --clip-backend triton
run cb_clean_torch    $PY examples/train_dpsgd.py ${TD/--stop-at-step 6/--stop-at-step 8}
run cb_clean_triton   $PY examples/train_dpsgd.py ${TD/--stop-at-step 6/--stop-at-step 8} --clip-backend triton
echo CBDONE

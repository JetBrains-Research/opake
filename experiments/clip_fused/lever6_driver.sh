#!/bin/bash
# Crossover: do Opake's kernels pay off once each call covers more examples (mb=4)?
set -u
cd ~/opaque; mkdir -p runs/lv
until grep -q LV5DONE ~/lever5_driver.log 2>/dev/null; do sleep 20; done
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --clip-backend triton --microbatch-size 4"
DT="--model-name Qwen/Qwen2.5-Coder-7B --dataset JetBrains/KStack --dataset-text-field content --dtype bfloat16 --lora-r 16 --lora-alpha 16 --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-eval-on-start --use-performance-kernels --target-epsilon 3.0 --no-wandb --log-steps 1 --clip-backend triton --microbatch-size 4"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1 ENVS=$2; shift 2
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/lv/$P.smi & local SMI=$!
  env $ENVS timeout 1500 $PY $W "$@" > runs/lv/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run lv_kern_mb4 "CB_X=0"       examples/train_dpsgd.py $TD --memory-profile runs/lv/lv_kern_mb4 $MP
run dt_mb4      "CB_X=0"       examples/train_dpsgd_trainer.py $DT --output-dir /tmp/dt_mb4
run dt_np_mb4   "CB_NO_PEFT=1" examples/train_dpsgd_trainer.py $DT --output-dir /tmp/dt_np_mb4
echo LV6DONE

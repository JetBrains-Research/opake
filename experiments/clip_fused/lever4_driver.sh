#!/bin/bash
# DPTrainer kernel profile + stock-PEFT A/B (kernel-targets report).
set -u
cd ~/opaque; mkdir -p runs/kp runs/lv
until grep -q LV3DONE ~/lever3_driver.log 2>/dev/null; do sleep 20; done
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
DT="--model-name Qwen/Qwen2.5-Coder-7B --dataset JetBrains/KStack --dataset-text-field content --dtype bfloat16 --lora-r 16 --lora-alpha 16 --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-eval-on-start --use-performance-kernels --target-epsilon 3.0 --no-wandb --log-steps 1 --clip-backend triton --microbatch-size 2"
echo "=== kp_dt start $(date -u +%T)"
CB_KPROF=runs/kp/kp_dt CB_KPROF_STEPS="4:cuda,5:all" timeout 1500 $PY $W examples/train_dpsgd_trainer.py $DT --output-dir /tmp/kp_dt > runs/kp/kp_dt.log 2>&1
echo "=== kp_dt exit $? $(date -u +%T)"
for P in dt_base dt_np; do
  ENVS="CB_X=0"; [ $P = dt_np ] && ENVS="CB_NO_PEFT=1"
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/lv/$P.smi & SMI=$!
  env $ENVS timeout 1500 $PY $W examples/train_dpsgd_trainer.py $DT --output-dir /tmp/$P > runs/lv/$P.log 2>&1
  echo "=== $P exit $? $(date -u +%T)"; kill $SMI 2>/dev/null
done
echo LV4DONE

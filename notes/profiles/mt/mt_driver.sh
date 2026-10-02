#!/bin/bash
set -u
cd ~/opaque; mkdir -p runs
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --clip-backend triton"
DT="--model-name Qwen/Qwen2.5-Coder-7B --dataset JetBrains/KStack --dataset-text-field content --dtype bfloat16 --lora-r 16 --lora-alpha 16 --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-eval-on-start --use-performance-kernels --target-epsilon 3.0 --no-wandb --log-steps 1 --clip-backend triton"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1 MT=$2 CH=$3; shift 3
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/$P.smi & local SMI=$!
  CB_MT=$MT CB_VMAP_CHUNK=$CH $PY $W "$@" > runs/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
for K in eager kern; do
  KF=""; [ $K = eager ] && KF="--no-kernel-patches"
  run mt_${K}_pl_mb 0 0 examples/train_dpsgd.py $TD $KF --microbatch-size 2 --memory-profile runs/mt_${K}_pl_mb $MP
  run mt_${K}_mt_mb 1 0 examples/train_dpsgd.py $TD $KF --microbatch-size 2 --memory-profile runs/mt_${K}_mt_mb $MP
  run mt_${K}_pl_ch 0 2 examples/train_dpsgd.py $TD $KF --microbatch-size 0 --memory-profile runs/mt_${K}_pl_ch $MP
  run mt_${K}_mt_ch 1 2 examples/train_dpsgd.py $TD $KF --microbatch-size 0 --memory-profile runs/mt_${K}_mt_ch $MP
done
CL="${TD/--stop-at-step 6/--stop-at-step 8} --no-kernel-patches"
run mt_clean_pl_mb 0 0 examples/train_dpsgd.py $CL --microbatch-size 2
run mt_clean_mt_mb 1 0 examples/train_dpsgd.py $CL --microbatch-size 2
run mt_clean_mt_ch 1 2 examples/train_dpsgd.py $CL --microbatch-size 0
run mt_dt_pl_mb 0 0 examples/train_dpsgd_trainer.py $DT --microbatch-size 2 --output-dir /tmp/mt_dt_pl_mb
run mt_dt_mt_mb 1 0 examples/train_dpsgd_trainer.py $DT --microbatch-size 2 --output-dir /tmp/mt_dt_mt_mb
run mt_dt_mt_ch 1 2 examples/train_dpsgd_trainer.py $DT --microbatch-size 0 --output-dir /tmp/mt_dt_mt_ch
echo MTDONE

#!/bin/bash
# Fused Triton clip inside the graph + DPTrainer without the (uncapturable) fused linear CE.
set -u
cd ~/opaque; mkdir -p runs/cg
until grep -q CG3DONE ~/cg3_driver.log 2>/dev/null; do sleep 20; done
export WANDB_MODE=disabled
PY=.venv/bin/python; W=experiments/clip_fused/run_train_variant.py
echo "=== gates $(date -u +%T)"
$PY -m pytest packages/opake-engine/tests/clipping/test_clip_backend.py -q -p no:cacheprovider -m "cuda and not slow" 2>&1 | tail -1
$PY experiments/cuda_graph/check_cudagraph_small.py triton 2>&1 | grep -v Warning | tail -2
TD="--preset qwen-7b-kstack --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2"
DT="--model-name Qwen/Qwen2.5-Coder-7B --dataset JetBrains/KStack --dataset-text-field content --dtype bfloat16 --lora-r 16 --lora-alpha 16 --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --max-seq-len 512 --batch-size 16 --stop-at-step 6 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100 --eval-batch-size 2 --no-eval-on-start --use-performance-kernels --target-epsilon 3.0 --no-wandb --log-steps 1"
MP="--memory-profile-skip 3 --memory-profile-steps 2"
run() { local P=$1 ENVS=$2; shift 2
  echo "=== $P start $(date -u +%T)"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/cg/$P.smi & local SMI=$!
  env $ENVS timeout 1500 $PY $W "$@" > runs/cg/$P.log 2>&1; local rc=$?
  kill $SMI 2>/dev/null; echo "=== $P exit $rc $(date -u +%T)"
}
run cg_fused_check  "CB_CUDAGRAPH=1 CB_CUDAGRAPH_FUSED=1 CB_CUDAGRAPH_CHECK=1" examples/train_dpsgd.py ${TD/--stop-at-step 6/--stop-at-step 4} --no-kernel-patches --clip-backend triton --microbatch-size 2
run cg_fused_mb2    "CB_CUDAGRAPH=1 CB_CUDAGRAPH_FUSED=1"                      examples/train_dpsgd.py $TD --no-kernel-patches --clip-backend triton --microbatch-size 2 --memory-profile runs/cg/cg_fused_mb2 $MP
run cg_fused_kern_tf32_mb4 "CB_CUDAGRAPH=1 CB_CUDAGRAPH_FUSED=1 CB_TF32=1"     examples/train_dpsgd.py $TD --clip-backend triton --microbatch-size 4 --memory-profile runs/cg/cg_fused_kern_tf32_mb4 $MP
run dt_nfce_mb2     "CB_NO_FUSED_CE=1"                  examples/train_dpsgd_trainer.py $DT --clip-backend triton --microbatch-size 2 --output-dir /tmp/dt_nfce_mb2
run dt_cg_nfce_mb2  "CB_NO_FUSED_CE=1 CB_CUDAGRAPH=1 CB_CUDAGRAPH_FUSED=1" examples/train_dpsgd_trainer.py $DT --clip-backend triton --microbatch-size 2 --output-dir /tmp/dt_cg_nfce_mb2
run dt_nfce_mb4     "CB_NO_FUSED_CE=1"                  examples/train_dpsgd_trainer.py $DT --clip-backend triton --microbatch-size 4 --output-dir /tmp/dt_nfce_mb4
run dt_cg_nfce_mb4  "CB_NO_FUSED_CE=1 CB_CUDAGRAPH=1 CB_CUDAGRAPH_FUSED=1" examples/train_dpsgd_trainer.py $DT --clip-backend triton --microbatch-size 4 --output-dir /tmp/dt_cg_nfce_mb4
echo CG4DONE

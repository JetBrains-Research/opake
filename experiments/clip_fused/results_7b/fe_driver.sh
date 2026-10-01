#!/bin/bash
set -u
cd ~/opaque
mkdir -p runs
COMMON="--preset qwen-7b-kstack --max-seq-len 512 --microbatch-size 2 --batch-size 16 --num-train-samples 2000 --num-eval-samples 100 --eval-steps 100"
W=experiments/clip_fused/run_train_with_engine.py
PY=.venv/bin/python
stamp() { echo "=== $1 $(date -u +%T)"; }
stamp "R1 base memprofile start"
OPAKE_CLIP_ENGINE=stream $PY $W $COMMON --stop-at-step 6 --memory-profile runs/fe_base --memory-profile-skip 3 --memory-profile-steps 2 > runs/fe_base.log 2>&1; echo "R1 exit $?"
stamp "R2 fused memprofile start"
OPAKE_CLIP_ENGINE=fused $PY $W $COMMON --stop-at-step 6 --memory-profile runs/fe_fused --memory-profile-skip 3 --memory-profile-steps 2 > runs/fe_fused.log 2>&1; echo "R2 exit $?"
stamp "R3 base attribution start"
OPAKE_CLIP_ENGINE=stream OPAKE_SEAM_TIMING=1 $PY $W $COMMON --stop-at-step 6 > runs/fe_attr_base.log 2>&1; echo "R3 exit $?"
stamp "R4 fused attribution start"
OPAKE_CLIP_ENGINE=fused OPAKE_SEAM_TIMING=1 $PY $W $COMMON --stop-at-step 6 > runs/fe_attr_fused.log 2>&1; echo "R4 exit $?"
for e in stream fused; do
  stamp "clean $e start"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > runs/fe_clean_$e.smi & SMI=$!
  OPAKE_CLIP_ENGINE=$e $PY $W $COMMON --stop-at-step 8 > runs/fe_clean_$e.log 2>&1; echo "clean $e exit $?"
  kill $SMI
done
stamp ALLDONE

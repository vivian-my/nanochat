#!/bin/bash
# WeCo launch — ClimbMix v19: gpt-5.5 × 200 steps, full 9-feature bank + ablation priors, Modal.
# Re-uses the deployed nanochat-d8-lexical-eval app (same as v9/v10/v13).

cd /home/nvidia/yan/nanochat
source .venv/bin/activate

export WECO_RUN_PREFIX="climbmix_v19"

weco run \
    --sources data_engineering/data_select_climbmix_v19.py \
    --eval-command 'bash weco/eval/weco_eval_climbmix_v19.sh' \
    --metric val_bpb \
    --goal minimize \
    --steps 200 \
    --model gpt-5.5 \
    --eval-timeout 2400 \
    --save-logs \
    --additional-instructions weco/instructions/weco_instructions_climbmix_v7.md

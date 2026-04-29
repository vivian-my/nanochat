#!/bin/bash
# WeCo launch — ClimbMix v17: gpt-5.5 × 200 steps, full 9-feature bank, NO priors, Modal.
# Re-uses the deployed nanochat-d8-lexical-eval app (same as v9/v10/v13).
# 3 seeds per trial → 3 H100s in parallel per WECO step.

cd /home/nvidia/yan/nanochat
source .venv/bin/activate

export WECO_RUN_PREFIX="climbmix_v17"

weco run \
    --sources data_engineering/data_select_climbmix_v17.py \
    --eval-command 'bash weco/eval/weco_eval_climbmix_v17.sh' \
    --metric val_bpb \
    --goal minimize \
    --steps 200 \
    --model gpt-5.5 \
    --eval-timeout 2400 \
    --save-logs \
    --additional-instructions weco/instructions/weco_instructions_climbmix_v17.md

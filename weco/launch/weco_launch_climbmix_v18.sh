#!/bin/bash
# WeCo launch — ClimbMix v18: gpt-5.5 × 200 steps, feature_construct + selection on Modal.
#
# Prerequisites (run once):
#   /home/nvidia/yan/modal_env/bin/modal deploy \
#       data_engineering/modal_train_d8_features.py

cd /home/nvidia/yan/nanochat
source .venv/bin/activate

export WECO_RUN_PREFIX="climbmix_v18"

weco run \
    --sources data_engineering/data_select_climbmix_v18.py \
    --eval-command 'bash weco/eval/weco_eval_climbmix_v18.sh' \
    --metric val_bpb \
    --goal minimize \
    --steps 200 \
    --model gpt-5.5 \
    --eval-timeout 3600 \
    --save-logs \
    --additional-instructions weco/instructions/weco_instructions_climbmix_v18.md

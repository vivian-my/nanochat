#!/bin/bash
# WeCo launch — ClimbMix v16: cheap-features-only on 100-shard pool, ratio=10.5, via Modal.
#
# Prerequisites (run once):
#   1. Upload 100-shard pool + cheap-feature meta to a new Modal volume:
#        /home/nvidia/yan/modal_env/bin/modal run \
#            data_engineering/modal_upload_climbmix_100pool.py
#   2. Deploy the Modal training app:
#        /home/nvidia/yan/modal_env/bin/modal deploy \
#            data_engineering/modal_train_d8_cheap_100.py

cd /home/nvidia/yan/nanochat
source .venv/bin/activate

export WECO_RUN_PREFIX="climbmix_v16"

weco run \
    --sources data_engineering/data_select_climbmix_v16.py \
    --eval-command 'bash weco/eval/weco_eval_climbmix_v16.sh' \
    --metric val_bpb \
    --goal minimize \
    --steps 200 \
    --model gpt-5.4 \
    --eval-timeout 2400 \
    --save-logs \
    --additional-instructions weco/instructions/weco_instructions_climbmix_v16.md

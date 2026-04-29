#!/bin/bash
# WeCo launch — ClimbMix v15: 100-shard pool, cheap features only, ratio=20, GPUs 4-7.
#
# One-time setup (run manually first):
#   mkdir -p ~/.cache/nanochat_v15/base_checkpoints /dev/shm/climbmix_v15_run
#   ln -sfn /dev/shm/climbmix_v15_run                            ~/.cache/nanochat_v15/base_data_climbmix
#   ln -sfn /home/nvidia/.cache/nanochat/base_data_climbmix_full ~/.cache/nanochat_v15/base_data_climbmix_full
#   ln -sfn /home/nvidia/.cache/nanochat/tokenizer              ~/.cache/nanochat_v15/tokenizer
#   ln -sfn /home/nvidia/.cache/nanochat/eval_bundle            ~/.cache/nanochat_v15/eval_bundle
#   ln -sfn /home/nvidia/.cache/nanochat/annotation             ~/.cache/nanochat_v15/annotation

cd /home/nvidia/yan/nanochat
source .venv/bin/activate

export WECO_RUN_PREFIX="climbmix_v15"

weco run \
    --sources data_engineering/data_select_climbmix_v15.py \
    --eval-command 'bash weco/eval/weco_eval_climbmix_v15.sh' \
    --metric val_bpb \
    --goal minimize \
    --steps 200 \
    --model gpt-5.4 \
    --eval-timeout 4800 \
    --save-logs \
    --additional-instructions weco/instructions/weco_instructions_climbmix_v15.md

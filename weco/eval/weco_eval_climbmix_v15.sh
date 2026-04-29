#!/bin/bash
# v15 eval wrapper — uses GPUs 4-7 + $HOME/.cache/nanochat_v15.
set -u
cd /home/nvidia/yan/nanochat
source .venv/bin/activate
exec .venv/bin/python -m data_engineering.weco_eval_climbmix_v15

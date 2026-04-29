#!/bin/bash
# v11 eval wrapper — uses GPUs 0-3 + $HOME/.cache/nanochat_v11.
set -u
cd /home/nvidia/yan/nanochat
source .venv/bin/activate
exec .venv/bin/python -m data_engineering.weco_eval_climbmix_v11

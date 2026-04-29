#!/bin/bash
# v16 eval wrapper — Modal-based d8 (cheap features, 100-shard pool, ratio=10.5).
set -u
cd /home/nvidia/yan/nanochat
source .venv/bin/activate
exec /home/nvidia/yan/modal_env/bin/python -m data_engineering.weco_eval_climbmix_v16

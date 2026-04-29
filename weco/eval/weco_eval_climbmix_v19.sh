#!/bin/bash
# v19 eval wrapper — Modal-based d8 (full 9 features, gpt-5.5, ablation priors).
set -u
cd /home/nvidia/yan/nanochat
source .venv/bin/activate
exec /home/nvidia/yan/modal_env/bin/python -m data_engineering.weco_eval_climbmix_v19

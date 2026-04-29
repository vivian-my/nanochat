#!/bin/bash
# v18 eval wrapper — Modal-based d8 + feature_construct (composites + ≤3 lexicals).
set -u
cd /home/nvidia/yan/nanochat
source .venv/bin/activate
exec /home/nvidia/yan/modal_env/bin/python -m data_engineering.weco_eval_climbmix_v18

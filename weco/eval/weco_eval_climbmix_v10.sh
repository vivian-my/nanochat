#!/bin/bash
# v10 eval wrapper — Modal-based d8 training (no local GPU usage).
# Requires deployed Modal app (see modal_train_d8_lexical.py header).
set -u
cd /home/nvidia/yan/nanochat
# Use modal_env so the modal SDK is available
exec /home/nvidia/yan/modal_env/bin/python -m data_engineering.weco_eval_climbmix_v10

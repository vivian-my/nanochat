"""WeCo eval v16: Modal-based d8 selection eval (cheap-features-only on 100 shards).

Identical training config to v9/v10 (depth=8, FP8, NUM_ITERS=3357 → ratio=10.5),
just on the 100-shard pool (8.46M docs) with cheap features only:
topic_id, format_id, avg_distinct_ngram_bpe (+ per-n), doc_tokens, doc_chars.

Each iteration:
  1. Load `select_docs` from data_engineering/data_select_climbmix_v16
  2. Compute selection (BUDGET=880,000 indices)
  3. Dispatch 4 d8 trainings on Modal (H100:1 each) in parallel
  4. Collect val_bpb per seed, print mean + std for WECO

Requires the Modal app to be deployed first:
  /home/nvidia/yan/modal_env/bin/modal deploy \\
      /home/nvidia/yan/nanochat/data_engineering/modal_train_d8_cheap_100.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/nvidia/yan/nanochat")
sys.path.insert(0, str(REPO))

import modal

PENALTY_BPB = 9.9999
N_SEEDS = 4

run_prefix = os.environ.get("WECO_RUN_PREFIX", "climbmix_v16")
STATE_FILE = REPO / f".weco_iteration_{run_prefix}"


def get_iteration():
    n = int(STATE_FILE.read_text().strip()) if STATE_FILE.exists() else 0
    STATE_FILE.write_text(str(n + 1))
    return n


def log(msg):
    print(f"[weco_eval_v16] {msg}", flush=True)


def validate_selection(idx, budget):
    from data_engineering.data_select_climbmix_v16 import N_TOTAL
    assert isinstance(idx, np.ndarray), f"must return np.ndarray, got {type(idx)}"
    assert idx.dtype == np.int64, f"dtype must be int64, got {idx.dtype}"
    assert idx.shape == (budget,), f"shape must be ({budget},), got {idx.shape}"
    if len(np.unique(idx)) != budget:
        raise ValueError(f"selection has duplicates ({budget - len(np.unique(idx))})")
    if idx.min() < 0 or idx.max() >= N_TOTAL:
        raise ValueError(f"indices out of range [0,{N_TOTAL}): min={idx.min()} max={idx.max()}")


def main():
    iteration = get_iteration()
    log(f"=== iteration {iteration} | {run_prefix}_step{iteration:02d} ===")

    from data_engineering.data_select_climbmix_v16 import select_docs, BUDGET
    t0 = time.time()
    idx = select_docs(BUDGET, seed=42)
    log(f"select_docs returned {len(idx):,} docs in {time.time()-t0:.1f}s")
    try:
        validate_selection(idx, BUDGET)
    except Exception as e:
        log(f"SELECTION INVALID: {e}")
        print(f"val_bpb: {PENALTY_BPB:.6f}")
        return

    train_d8 = modal.Function.from_name("nanochat-d8-cheap-100-eval", "train_d8")
    log(f"Dispatching {N_SEEDS} d8 trainings on Modal (H100:1 each)…")
    t1 = time.time()
    idx_list = idx.tolist()
    futures = [train_d8.spawn(seed=k, selection_indices=idx_list) for k in range(N_SEEDS)]
    results = [f.get() for f in futures]
    wall_min = (time.time() - t1) / 60

    bpbs_all = [r.get("val_bpb") for r in results]
    bpbs = [b for b in bpbs_all if b is not None]
    log(f"per-seed bpb: {bpbs_all}  (wall={wall_min:.1f}m)")

    if len(bpbs) < N_SEEDS:
        log(f"only {len(bpbs)}/{N_SEEDS} seeds produced val_bpb — penalty")
        print(f"val_bpb: {PENALTY_BPB:.6f}")
        return

    mean = float(np.mean(bpbs))
    std  = float(np.std(bpbs))
    log(f"mean={mean:.6f}  std={std:.6f}  n_docs={len(idx):,}")
    print(f"val_bpb: {mean:.6f}")
    print(f"val_bpb_std: {std:.6f}")


if __name__ == "__main__":
    main()

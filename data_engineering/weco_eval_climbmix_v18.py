"""WeCo eval v18: feature_construct + selection on Modal.

Each iteration:
  1. Read the current selector source from data_select_climbmix_v18.py.
  2. Validate that feature_specs() and select_docs() exist locally and at least
     don't crash on a dry run with the empty BASE/COMPOSITES/LEXICALS dicts
     (mostly a syntax check; lexicals can't be evaluated locally).
  3. Dispatch 3 d8 trainings on Modal (H100:1 each), passing the source string.
     The Modal harness handles feature_construct, lexical caching, training.
  4. Collect val_bpb per seed, print mean + std for WECO.

Modal app must be deployed first:
  /home/nvidia/yan/modal_env/bin/modal deploy \\
      /home/nvidia/yan/nanochat/data_engineering/modal_train_d8_features.py
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
N_SEEDS = 3

run_prefix = os.environ.get("WECO_RUN_PREFIX", "climbmix_v18")
STATE_FILE = REPO / f".weco_iteration_{run_prefix}"
SELECTOR_PATH = REPO / "data_engineering" / "data_select_climbmix_v18.py"


def get_iteration():
    n = int(STATE_FILE.read_text().strip()) if STATE_FILE.exists() else 0
    STATE_FILE.write_text(str(n + 1))
    return n


def log(msg):
    print(f"[weco_eval_v18] {msg}", flush=True)


def _local_syntax_check(source: str):
    """Compile the source and verify required symbols exist. Don't run lexicals."""
    try:
        compile(source, str(SELECTOR_PATH), "exec")
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    ns: dict = {}
    try:
        exec(compile(source, str(SELECTOR_PATH), "exec"), ns)
    except Exception as e:
        return False, f"import-time exec failed: {e}"
    for sym in ("BUDGET", "N_TOTAL", "feature_specs", "select_docs"):
        if sym not in ns:
            return False, f"missing symbol: {sym}"
    if not callable(ns["feature_specs"]) or not callable(ns["select_docs"]):
        return False, "feature_specs/select_docs must be callable"
    try:
        comps, lexs = ns["feature_specs"]()
        comps = list(comps or [])
        lexs = list(lexs or [])
    except Exception as e:
        return False, f"feature_specs() raised: {e}"
    return True, f"feature_specs OK: {len(comps)} composites, {len(lexs)} lexicals (cap=3)"


def main():
    iteration = get_iteration()
    log(f"=== iteration {iteration} | {run_prefix}_step{iteration:02d} ===")

    src = SELECTOR_PATH.read_text()
    ok, info = _local_syntax_check(src)
    log(info)
    if not ok:
        log(f"SELECTOR INVALID: {info}")
        print(f"val_bpb: {PENALTY_BPB:.6f}")
        return

    train_d8 = modal.Function.from_name("nanochat-d8-features-eval", "train_d8")
    log(f"Dispatching {N_SEEDS} d8 trainings on Modal (H100:1 each)…")
    t1 = time.time()
    futures = [train_d8.spawn(seed=k, selector_source=src) for k in range(N_SEEDS)]
    results = [f.get() for f in futures]
    wall_min = (time.time() - t1) / 60

    bpbs_all = [r.get("val_bpb") for r in results]
    log(f"per-seed bpb: {bpbs_all}  (wall={wall_min:.1f}m)")
    for r in results:
        log(f"  seed {r['seed']}: composites={r.get('composites_used', [])} "
            f"lexicals={r.get('lexicals_used', [])} "
            f"lex_compute_min={r.get('lexical_compute_min', 0):.1f}")

    bpbs = [b for b in bpbs_all if b is not None]
    if len(bpbs) < N_SEEDS:
        log(f"only {len(bpbs)}/{N_SEEDS} seeds produced val_bpb — penalty")
        print(f"val_bpb: {PENALTY_BPB:.6f}")
        return

    mean = float(np.mean(bpbs))
    std  = float(np.std(bpbs))
    log(f"mean={mean:.6f}  std={std:.6f}  n_docs={results[0].get('n_docs', '?')}")
    print(f"val_bpb: {mean:.6f}")
    print(f"val_bpb_std: {std:.6f}")


if __name__ == "__main__":
    main()

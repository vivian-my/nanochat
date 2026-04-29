"""WeCo eval v15: climbmix-v15 data selection on d8, 4 seeds in parallel on GPUs 4-7.

100-shard pool (8.46 M docs), cheap features only, d8 trained at target_param_ratio=20
(~838 M tokens per seed, NUM_ITERS=6394).

Differences vs v12 (the previous local-GPU eval):
  - 100-shard pool (meta_100shards), N_TOTAL=8,460,288
  - BUDGET=1,676,000 (vs 879,999) — ~19.8% keep
  - RATIO=20 (vs 10.5) → NUM_ITERS=6394 (vs 3357), ~2× wallclock per seed
  - GPUs 4..7 (v12 uses 0..3)
  - NANOCHAT_BASE_DIR=~/.cache/nanochat_v15
  - master_port = 36600 + seed
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/nvidia/yan/nanochat")
sys.path.insert(0, str(REPO))

ROOT = Path("/data/cache/nanochat")
POOL_DIR = ROOT / "base_data_climbmix_full"
VAL_SHARD = ROOT / "base_data_climbmix_full" / "shard_06542.parquet"

RUN_DIR = Path("/dev/shm/climbmix_v15_run")
NC_BASE = Path(os.path.expanduser("~/.cache/nanochat_v15"))
DATA_LINK = NC_BASE / "base_data_climbmix"
CKPT_BASE = NC_BASE / "base_checkpoints"

TOTAL_BATCH_SIZE = 131072
MAX_SEQ_LEN      = 1024
DEVICE_BATCH     = 128
DEPTH            = 8
RATIO            = 20.0
SCALING_PARAMS   = 41.9e6
TARGET_TOKENS    = int(SCALING_PARAMS * RATIO)                    # 838,000,000
NUM_ITERS        = int(os.environ.get("NUM_ITERS_OVERRIDE",
                                      TARGET_TOKENS // TOTAL_BATCH_SIZE))   # 6394

N_SEEDS          = 4
GPUS             = [4, 5, 6, 7]
TRAIN_TIMEOUT_S  = 3600

PENALTY_BPB      = 9.9999

run_prefix = os.environ.get("WECO_RUN_PREFIX", "climbmix_v15")
STATE_FILE = REPO / f".weco_iteration_{run_prefix}"


def get_iteration():
    n = int(STATE_FILE.read_text().strip()) if STATE_FILE.exists() else 0
    STATE_FILE.write_text(str(n + 1))
    return n


def log(msg):
    print(f"[weco_eval_v15] {msg}", flush=True)


def validate_selection(idx, budget):
    from data_engineering.data_select_climbmix_v15 import N_TOTAL
    assert isinstance(idx, np.ndarray), f"must return np.ndarray, got {type(idx)}"
    assert idx.dtype == np.int64, f"dtype must be int64, got {idx.dtype}"
    assert idx.shape == (budget,), f"shape must be ({budget},), got {idx.shape}"
    if len(np.unique(idx)) != budget:
        raise ValueError(f"selection has duplicates ({budget - len(np.unique(idx))})")
    if idx.min() < 0 or idx.max() >= N_TOTAL:
        raise ValueError(f"indices out of range [0,{N_TOTAL}): min={idx.min()} max={idx.max()}")


def setup_symlink():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_LINK.is_symlink():
        raise RuntimeError(
            f"expected {DATA_LINK} to be a symlink → {RUN_DIR}. "
            f"Run: ln -s {RUN_DIR} {DATA_LINK}"
        )
    log(f"symlink OK: {DATA_LINK} -> {RUN_DIR}")


def materialize(idx):
    import pyarrow as pa
    import pyarrow.parquet as pq
    for p in RUN_DIR.iterdir():
        p.unlink()

    offsets_path = ROOT / "annotation/annotation_climbmix_full_v2/meta_100shards/shard_offsets.json"
    entries = json.loads(offsets_path.read_text())

    global_idx = np.asarray(idx, dtype=np.int64)
    buckets = {}
    for entry in entries:
        off = entry["offset"]; cnt = entry["count"]
        mask = (global_idx >= off) & (global_idx < off + cnt)
        rows = (global_idx[mask] - off).astype(np.int64)
        if len(rows):
            buckets[entry["shard_idx"]] = rows

    t0 = time.time()
    parts = []
    for shard_i in sorted(buckets):
        src = POOL_DIR / f"shard_{shard_i:05d}.parquet"
        table = pq.read_table(src, columns=["text"])
        parts.append(table.take(pa.array(buckets[shard_i])))
    train_table = pa.concat_tables(parts)
    n_docs = train_table.num_rows
    dt_read = time.time() - t0

    t1 = time.time()
    n_out = 8
    per = (n_docs + n_out - 1) // n_out
    for i in range(n_out):
        lo, hi = i * per, min((i + 1) * per, n_docs)
        if lo >= hi: break
        sub = train_table.slice(lo, hi - lo)
        pq.write_table(sub, RUN_DIR / f"shard_{i:05d}.parquet", compression="snappy")
    shutil.copy(VAL_SHARD, RUN_DIR / "shard_99999.parquet")
    dt_write = time.time() - t1
    log(f"materialized {n_docs:,} docs into {n_out} train parquets + 1 val  "
        f"(read={dt_read:.1f}s, write={dt_write:.1f}s)")
    return n_docs


def launch_one(seed, gpu, iteration):
    tag = f"{run_prefix}_step{iteration:02d}_s{seed}"
    ckpt_dir = CKPT_BASE / tag
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    log_path = REPO / f"weco/log/{run_prefix}/step{iteration:02d}_s{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "NANOCHAT_SEED":        str(seed),
        "NANOCHAT_BASE_DIR":    str(NC_BASE),
        "WANDB_MODE":           "disabled",
    })
    cmd = [
        str(REPO / ".venv/bin/torchrun"),
        "--standalone", "--nproc_per_node=1", f"--master_port={36600+seed}",
        "-m", "scripts.base_train", "--",
        f"--depth={DEPTH}",
        f"--device-batch-size={DEVICE_BATCH}",
        f"--max-seq-len={MAX_SEQ_LEN}",
        f"--total-batch-size={TOTAL_BATCH_SIZE}",
        f"--num-iterations={NUM_ITERS}",
        "--fp8",
        f"--model-tag={tag}",
        f"--run={tag}",
        "--eval-every=999999",
        "--core-metric-every=-1",
        "--sample-every=-1",
        "--save-every=-1",
        "--eval-tokens=2097152",
    ]
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
    return proc, fh, log_path


def parse_final_bpb(log_path):
    txt = log_path.read_text()
    bpbs = re.findall(r"Validation bpb:\s*([\d.]+)", txt)
    return float(bpbs[-1]) if bpbs else None


def main():
    iteration = get_iteration()
    log(f"=== iteration {iteration} | {run_prefix}_step{iteration:02d} ===")

    from data_engineering.data_select_climbmix_v15 import select_docs, BUDGET
    t0 = time.time()
    idx = select_docs(BUDGET, seed=42)
    log(f"select_docs returned {len(idx):,} docs in {time.time()-t0:.1f}s")
    try:
        validate_selection(idx, BUDGET)
    except Exception as e:
        log(f"SELECTION INVALID: {e}")
        print(f"val_bpb: {PENALTY_BPB:.6f}")
        return

    try:
        setup_symlink()
        n_docs = materialize(idx)
    except Exception as e:
        log(f"MATERIALIZE FAILED: {e}")
        print(f"val_bpb: {PENALTY_BPB:.6f}")
        raise

    log(f"launching {N_SEEDS} d8 trainings in parallel on GPUs {GPUS}  "
        f"(depth={DEPTH}, ratio={RATIO}, iters={NUM_ITERS}, "
        f"tokens={NUM_ITERS*TOTAL_BATCH_SIZE:,})")
    procs = []
    for k in range(N_SEEDS):
        p, fh, lp = launch_one(seed=k, gpu=GPUS[k], iteration=iteration)
        procs.append((k, p, fh, lp))
    t_train = time.time()
    for k, p, fh, lp in procs:
        rc = p.wait(timeout=TRAIN_TIMEOUT_S); fh.close()
        log(f"seed {k}: exit={rc}  log={lp}")
    train_min = (time.time() - t_train) / 60
    bpbs = [parse_final_bpb(lp) for _, _, _, lp in procs]
    ok = [b for b in bpbs if b is not None]
    if len(ok) < N_SEEDS:
        log(f"only {len(ok)}/{N_SEEDS} seeds produced val_bpb — penalty")
        print(f"val_bpb: {PENALTY_BPB:.6f}")
        return
    mean = float(np.mean(ok)); std = float(np.std(ok))
    log(f"val_bpb per seed: {', '.join(f'{b:.6f}' for b in bpbs)}")
    log(f"wall={train_min:.2f}m  mean={mean:.6f}  std={std:.6f}  n_docs={n_docs:,}")
    print(f"val_bpb: {mean:.6f}")
    print(f"val_bpb_std: {std:.6f}")


if __name__ == "__main__":
    main()

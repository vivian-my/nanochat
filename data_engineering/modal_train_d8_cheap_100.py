"""Modal app for d8 training, used by WECO v16 (cheap-features on 100 shards).

DEPLOY ONCE with:
    /home/nvidia/yan/modal_env/bin/modal deploy \\
        /home/nvidia/yan/nanochat/data_engineering/modal_train_d8_cheap_100.py

Then the WECO eval script invokes the deployed function via:
    train_d8 = modal.Function.from_name("nanochat-d8-cheap-100-eval", "train_d8")
    futures = [train_d8.spawn(seed=k, selection_indices=idx.tolist()) for k in range(4)]
    results = [f.get() for f in futures]

Each call: H100:1, ~10–12 min per seed. Reads shards + meta from the
nanochat-climbmix-100pool volume (no HuggingFace download).

Identical training config to v9/v10 (depth=8, ratio=10.5, NUM_ITERS=3357),
just on the 100-shard pool with cheap features available in /pool/meta.
"""
import os

import modal

VOLUME_NAME = "nanochat-climbmix-100pool"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "numpy>=1.26",
        "pyarrow",
        "tiktoken",
        "pyyaml",
        "regex",
        "wandb",
        "transformers",
        "rustbpe",
        "kernels>=0.11.7",
        "scipy",
        "tabulate",
        "zstandard",
        "psutil",
    )
    .add_local_dir("/home/nvidia/yan/nanochat/nanochat", remote_path="/root/nanochat/nanochat")
    .add_local_dir("/home/nvidia/yan/nanochat/scripts",  remote_path="/root/nanochat/scripts")
    .add_local_dir("/home/nvidia/yan/nanochat/tasks",    remote_path="/root/nanochat/tasks")
    .add_local_dir(os.path.expanduser("~/.cache/nanochat/tokenizer"),
                   remote_path="/root/.cache/nanochat/tokenizer")
    .add_local_dir(os.path.expanduser("~/.cache/nanochat/eval_bundle"),
                   remote_path="/root/.cache/nanochat/eval_bundle")
)

vol = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("nanochat-d8-cheap-100-eval", image=image)


@app.function(
    gpu="H100:1",
    timeout=2400,
    memory=65536,
    volumes={"/pool": vol},
)
def train_d8(seed: int, selection_indices: list):
    """Materialize selection from /pool/shards/, train d8, return val_bpb.

    Args:
      seed: training rng seed (0,1,2,3 typically)
      selection_indices: list of int64 global doc ids (length BUDGET=880,000)

    Returns dict with {seed, val_bpb, exit, wall_min}.
    """
    import re
    import shutil
    import subprocess
    import time

    import json
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.chdir("/root/nanochat")
    os.environ["HOME"] = "/root"
    os.environ["NANOCHAT_BASE_DIR"] = "/root/.cache/nanochat"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["NANOCHAT_SEED"] = str(seed)

    POOL_SHARDS = "/pool/shards"
    POOL_META   = "/pool/meta"
    DATA_DIR    = "/root/.cache/nanochat/base_data_climbmix"
    os.makedirs(DATA_DIR, exist_ok=True)

    indices = np.asarray(selection_indices, dtype=np.int64)
    print(f"[seed {seed}] materializing {len(indices):,} docs", flush=True)

    shard_offsets = json.loads(open(f"{POOL_META}/shard_offsets.json").read())

    t0 = time.time()
    all_docs = []
    for entry in sorted(shard_offsets, key=lambda e: e["shard_idx"]):
        si  = entry["shard_idx"]; off = entry["offset"]; cnt = entry["count"]
        mask = (indices >= off) & (indices < off + cnt)
        if not mask.any():
            continue
        local_idx = (indices[mask] - off).astype(np.int64)

        src = f"{POOL_SHARDS}/shard_{si:05d}.parquet"
        pf = pq.ParquetFile(src)
        texts = []
        for rg_i in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_i)
            texts.extend(rg.column("text").to_pylist())
        for li in local_idx:
            t = texts[int(li)]
            if isinstance(t, bytes):
                t = t.decode("utf-8", errors="replace")
            all_docs.append(t)
    print(f"[seed {seed}] materialized {len(all_docs):,} docs in {time.time()-t0:.0f}s", flush=True)

    DOCS_PER_SHARD = 86016
    n_shards = (len(all_docs) + DOCS_PER_SHARD - 1) // DOCS_PER_SHARD
    for i in range(n_shards):
        s = i * DOCS_PER_SHARD
        e = min(s + DOCS_PER_SHARD, len(all_docs))
        path = os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")
        w = pq.ParquetWriter(path, pa.schema([("text", pa.string())]))
        w.write_table(pa.table({"text": all_docs[s:e]}))
        w.close()
    shutil.copy(f"{POOL_SHARDS}/shard_06542.parquet",
                os.path.join(DATA_DIR, f"shard_{n_shards:05d}.parquet"))
    print(f"[seed {seed}] wrote {n_shards+1} parquets, starting d8", flush=True)
    del all_docs

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        "-m", "scripts.base_train", "--",
        "--depth=8",
        "--device-batch-size=128",
        "--max-seq-len=1024",
        "--total-batch-size=131072",
        "--num-iterations=3357",
        "--fp8",
        f"--run=v16_modal_s{seed}",
        f"--model-tag=v16_modal_s{seed}",
        "--eval-every=999999",
        "--core-metric-every=-1",
        "--sample-every=-1",
        "--save-every=-1",
        "--eval-tokens=2097152",
    ]
    t1 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=2200)
    out = res.stdout + res.stderr
    bpbs = re.findall(r"Validation bpb:\s*([\d.]+)", out)
    val_bpb = float(bpbs[-1]) if bpbs else None

    wall_min = (time.time() - t1) / 60
    print(f"[seed {seed}] training {wall_min:.1f}m exit={res.returncode} val_bpb={val_bpb}", flush=True)
    if val_bpb is None or res.returncode not in (0,):
        if val_bpb is None:
            print(f"[seed {seed}] LAST 40 LINES:")
            for line in out.splitlines()[-40:]:
                print(f"  [seed {seed}]  {line}")
    return {"seed": seed, "val_bpb": val_bpb, "exit": res.returncode, "wall_min": wall_min}

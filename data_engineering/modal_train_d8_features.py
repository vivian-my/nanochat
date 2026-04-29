"""Modal app for d8 training, used by WECO v18 (feature_construct + selection).

DEPLOY ONCE with:
    /home/nvidia/yan/modal_env/bin/modal deploy \\
        /home/nvidia/yan/nanochat/data_engineering/modal_train_d8_features.py

Then the WECO eval script invokes the deployed function via:
    train_d8 = modal.Function.from_name("nanochat-d8-features-eval", "train_d8")
    futures = [train_d8.spawn(seed=k, selector_source=src) for k in range(3)]

Pre-train phase (NEW vs modal_train_d8_lexical):
  1. Receives the selector source as a string (rather than just selection_indices).
  2. Loads the 9 base features from /pool/meta into BASE dict.
  3. exec()s the selector source in a controlled namespace.
  4. Calls feature_specs() → (composites, lexicals).
  5. Caps lexicals at 3 (drop overflow with a log line).
  6. Computes composites from BASE.
  7. For each lexical: source-hash check on /pool/meta_dynamic/<name>__<hash>.npy;
     if missing, run the per-doc fn over all 4.485 M docs in parallel and persist.
  8. Loads cached lexical arrays into LEXICALS dict.
  9. Calls select_docs(BUDGET, seed) → indices.
 10. Materialize, train d8 (existing logic), return val_bpb.

Each call: H100:1, ~12-15 min training + up to ~3 min × 3 lexicals if all are
fresh (most should be cache hits across trials).
"""
import os

import modal

VOLUME_NAME = "nanochat-climbmix-53pool"

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
app = modal.App("nanochat-d8-features-eval", image=image)

BASE_FEATURE_FILES = {
    "topic_id":             "topic_id.npy",
    "format_id":            "format_id.npy",
    "n_rsteps":             "n_rsteps.npy",
    "n_rerrors":            "n_rerrors.npy",
    "n_factual":            "n_factual.npy",
    "logppl_qwen":          "logppl_qwen.npy",
    "logppl_d8_ref":        "logppl_d8_ref.npy",
    "avg_distinct_ngram_bpe": "avg_distinct_ngram_bpe.npy",
    "doc_tokens":           "doc_tokens.npy",
    # plus utilities — same dir
    "doc_chars":            "doc_chars.npy",
    "ok":                   "ok.npy",
    "distinct_1gram_bpe":   "distinct_1gram_bpe.npy",
    "distinct_2gram_bpe":   "distinct_2gram_bpe.npy",
    "distinct_3gram_bpe":   "distinct_3gram_bpe.npy",
    "distinct_4gram_bpe":   "distinct_4gram_bpe.npy",
    "distinct_5gram_bpe":   "distinct_5gram_bpe.npy",
}

MAX_LEXICALS = 3
DYNAMIC_CACHE = "/pool/meta_dynamic"


@app.function(
    gpu="H100:1",
    timeout=2400,
    memory=65536,
    volumes={"/pool": vol},
)
def train_d8(seed: int, selector_source: str):
    """Run feature_construct → select_docs → train d8, return val_bpb.

    Args:
      seed: training rng seed (0,1,2 typically)
      selector_source: string content of the user's data_select_climbmix_v18.py
                       (sent each call so feature_specs reflects current LLM iteration)

    Returns dict {seed, val_bpb, exit, wall_min, n_docs, lexicals_used,
                   composites_used, lexical_compute_min}.
    """
    import hashlib
    import inspect
    import json
    import os as _os
    import re
    import shutil
    import subprocess
    import time
    import traceback
    from multiprocessing import Pool

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    _os.chdir("/root/nanochat")
    _os.environ["HOME"] = "/root"
    _os.environ["NANOCHAT_BASE_DIR"] = "/root/.cache/nanochat"
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["WANDB_MODE"] = "disabled"
    _os.environ["NANOCHAT_SEED"] = str(seed)

    POOL_SHARDS = "/pool/shards"
    POOL_META   = "/pool/meta"
    DATA_DIR    = "/root/.cache/nanochat/base_data_climbmix"
    _os.makedirs(DATA_DIR, exist_ok=True)
    _os.makedirs(DYNAMIC_CACHE, exist_ok=True)

    def log(msg):
        print(f"[seed {seed}] {msg}", flush=True)

    # ----------------------------------------------------------
    # Step 1: exec the selector source in a fresh module namespace.
    # ----------------------------------------------------------
    selector_ns = {"__name__": "data_select_climbmix_v18", "__file__": "<selector>"}
    try:
        exec(compile(selector_source, "<selector>", "exec"), selector_ns)
    except Exception as e:
        log(f"SELECTOR EXEC FAILED: {e}")
        traceback.print_exc()
        return {"seed": seed, "val_bpb": None, "exit": -1, "wall_min": 0,
                "error": f"exec: {e}"}

    BUDGET = selector_ns.get("BUDGET")
    N_TOTAL = selector_ns.get("N_TOTAL")
    feature_specs = selector_ns.get("feature_specs")
    select_docs = selector_ns.get("select_docs")
    if not (callable(feature_specs) and callable(select_docs) and BUDGET and N_TOTAL):
        log("SELECTOR MISSING REQUIRED SYMBOLS")
        return {"seed": seed, "val_bpb": None, "exit": -1, "wall_min": 0,
                "error": "missing feature_specs/select_docs/BUDGET/N_TOTAL"}

    # ----------------------------------------------------------
    # Step 2: load base features into BASE dict.
    # ----------------------------------------------------------
    BASE = {}
    for name, fname in BASE_FEATURE_FILES.items():
        path = f"{POOL_META}/{fname}"
        BASE[name] = np.load(path, mmap_mode="r")
    log(f"loaded {len(BASE)} base features")

    # ----------------------------------------------------------
    # Step 3: call feature_specs.
    # ----------------------------------------------------------
    try:
        composites_specs, lexicals_specs = feature_specs()
    except Exception as e:
        log(f"feature_specs FAILED: {e}")
        traceback.print_exc()
        return {"seed": seed, "val_bpb": None, "exit": -1, "wall_min": 0,
                "error": f"feature_specs: {e}"}

    composites_specs = list(composites_specs or [])
    lexicals_specs = list(lexicals_specs or [])
    if len(lexicals_specs) > MAX_LEXICALS:
        log(f"capping lexicals: {len(lexicals_specs)} → {MAX_LEXICALS} (dropped overflow)")
        lexicals_specs = lexicals_specs[:MAX_LEXICALS]

    log(f"feature_specs: {len(composites_specs)} composites, {len(lexicals_specs)} lexicals")

    # ----------------------------------------------------------
    # Step 4: compute composites (cheap, every call).
    # ----------------------------------------------------------
    COMPOSITES = {}
    for name, fn in composites_specs:
        try:
            arr = np.asarray(fn(BASE))
            if arr.shape != (N_TOTAL,):
                log(f"composite {name!r} bad shape {arr.shape}, skipping")
                continue
            COMPOSITES[name] = arr.astype(np.float32, copy=False)
        except Exception as e:
            log(f"composite {name!r} failed: {e}")
            continue

    # ----------------------------------------------------------
    # Step 5: lexicals — cache by name + source hash, compute if missing.
    # ----------------------------------------------------------
    LEXICALS = {}
    lexical_compute_secs = 0.0
    for name, fn in lexicals_specs:
        try:
            src = inspect.getsource(fn)
        except OSError:
            log(f"lexical {name!r}: cannot inspect source, skipping")
            continue
        h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
        cache_path = f"{DYNAMIC_CACHE}/{name}__{h}.npy"
        if _os.path.exists(cache_path):
            LEXICALS[name] = np.load(cache_path, mmap_mode="r")
            log(f"lexical {name!r}: cache HIT ({h})")
            continue

        log(f"lexical {name!r}: cache MISS ({h}), computing across {N_TOTAL:,} docs…")
        t0 = time.time()
        try:
            arr = _compute_lexical_over_pool(fn, N_TOTAL, POOL_SHARDS, POOL_META)
        except Exception as e:
            log(f"lexical {name!r} compute FAILED: {e}")
            traceback.print_exc()
            continue
        dt = time.time() - t0
        lexical_compute_secs += dt
        np.save(cache_path, arr)
        # Commit volume so other workers see the cache file.
        try:
            vol.commit()
        except Exception:
            pass
        LEXICALS[name] = arr
        log(f"lexical {name!r} computed in {dt:.0f}s, saved to {cache_path}")

    # ----------------------------------------------------------
    # Step 6: populate selector globals + run select_docs.
    # ----------------------------------------------------------
    selector_ns["BASE"] = BASE
    selector_ns["COMPOSITES"] = COMPOSITES
    selector_ns["LEXICALS"] = LEXICALS

    try:
        idx = select_docs(BUDGET, seed)
    except Exception as e:
        log(f"select_docs FAILED: {e}")
        traceback.print_exc()
        return {"seed": seed, "val_bpb": None, "exit": -1, "wall_min": 0,
                "error": f"select_docs: {e}"}

    idx = np.asarray(idx, dtype=np.int64)
    if idx.shape != (BUDGET,) or len(np.unique(idx)) != BUDGET or idx.min() < 0 or idx.max() >= N_TOTAL:
        log(f"INVALID SELECTION: shape={idx.shape} unique={len(np.unique(idx))} range=[{idx.min()},{idx.max()}]")
        return {"seed": seed, "val_bpb": None, "exit": -1, "wall_min": 0,
                "error": "invalid selection"}

    log(f"select_docs returned {len(idx):,} docs")

    # ----------------------------------------------------------
    # Step 7: materialize selected docs to /root/.cache/nanochat/base_data_climbmix.
    # ----------------------------------------------------------
    indices = idx
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
    log(f"materialized {len(all_docs):,} docs in {time.time()-t0:.0f}s")

    DOCS_PER_SHARD = 86016
    n_shards = (len(all_docs) + DOCS_PER_SHARD - 1) // DOCS_PER_SHARD
    for i in range(n_shards):
        s = i * DOCS_PER_SHARD
        e = min(s + DOCS_PER_SHARD, len(all_docs))
        path = _os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")
        w = pq.ParquetWriter(path, pa.schema([("text", pa.string())]))
        w.write_table(pa.table({"text": all_docs[s:e]}))
        w.close()
    shutil.copy(f"{POOL_SHARDS}/shard_06542.parquet",
                _os.path.join(DATA_DIR, f"shard_{n_shards:05d}.parquet"))
    log(f"wrote {n_shards+1} parquets, starting d8")
    del all_docs

    # ----------------------------------------------------------
    # Step 8: train d8 (same as v9/v10 — ratio 10.5, 3357 iters).
    # ----------------------------------------------------------
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        "-m", "scripts.base_train", "--",
        "--depth=8",
        "--device-batch-size=128",
        "--max-seq-len=1024",
        "--total-batch-size=131072",
        "--num-iterations=3357",
        "--fp8",
        f"--run=v18_modal_s{seed}",
        f"--model-tag=v18_modal_s{seed}",
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
    log(f"training {wall_min:.1f}m exit={res.returncode} val_bpb={val_bpb}")
    if val_bpb is None:
        log("LAST 40 LINES:")
        for line in out.splitlines()[-40:]:
            print(f"  [seed {seed}]  {line}", flush=True)

    return {
        "seed": seed,
        "val_bpb": val_bpb,
        "exit": res.returncode,
        "wall_min": wall_min,
        "n_docs": int(len(idx)),
        "composites_used": list(COMPOSITES.keys()),
        "lexicals_used": list(LEXICALS.keys()),
        "lexical_compute_min": lexical_compute_secs / 60,
    }


def _compute_lexical_over_pool(fn, n_total: int, pool_shards: str, pool_meta: str) -> "np.ndarray":
    """Run a per-doc text→float function across all docs in /pool/shards/.

    Single-process: reads each shard sequentially and applies fn doc-by-doc.
    Closure-based functions don't pickle cleanly to multiprocessing workers,
    so we trade parallelism for correctness. A typical regex-based lexical
    feature processes ~50K docs/sec, so 4.485 M docs ≈ 90 s total.
    """
    import json
    import time

    import numpy as np
    import pyarrow.parquet as pq

    shard_offsets = json.loads(open(f"{pool_meta}/shard_offsets.json").read())
    out = np.zeros(n_total, dtype=np.float32)

    for entry in shard_offsets:
        si = entry["shard_idx"]; off = entry["offset"]; cnt = entry["count"]
        t0 = time.time()
        table = pq.read_table(f"{pool_shards}/shard_{si:05d}.parquet", columns=["text"])
        texts = table.column("text").to_pylist()
        assert len(texts) == cnt, (si, len(texts), cnt)
        for i, text in enumerate(texts):
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            try:
                v = fn(text)
            except Exception:
                v = 0.0
            try:
                out[off + i] = float(v)
            except (TypeError, ValueError):
                out[off + i] = 0.0
        print(f"    [lex compute] shard {si:3d}: {cnt:,} docs in {time.time()-t0:.1f}s "
              f"({cnt/(time.time()-t0):.0f} docs/s)", flush=True)
    return out

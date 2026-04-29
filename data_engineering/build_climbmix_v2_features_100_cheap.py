"""Build cheap features (topic/format/ngram/length) for shards 0-99.

Output dir: /data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_100shards/

Cheap features (no LLM API, no GPU PPL):
  topic_id.npy, format_id.npy           — sliced from /data/jina_200shards/
  distinct_{1..5}gram_bpe.npy           — nanochat BPE n-gram diversity per doc
  avg_distinct_ngram_bpe.npy            — sum across n=1..5
  doc_tokens.npy                        — len(BPE-encoded prefix) per doc
  doc_chars.npy                         — len(text) per doc
  shard_offsets.json                    — list of {shard_idx, offset, count} (100 entries)
  topic_names.json, format_names.json   — copied from /data/jina_200shards/

LLM-annotation features (n_rsteps/n_rerrors/n_factual + ok mask) and PPL features
(logppl_qwen, logppl_d8_ref) are NOT built here — they require Gemini batch annotation
and GPU PPL scoring respectively.
"""
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/home/nvidia/yan/nanochat")

POOL = Path("/data/cache/nanochat/base_data_climbmix_full")
DDUDEK = Path("/data/jina_200shards")
OUT = Path("/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_100shards")
OUT.mkdir(parents=True, exist_ok=True)

N_SHARDS = 100
N_WORKERS = int(os.environ.get("N_WORKERS", "32"))
CHUNK_SIZE = 500

_TOK = None


def _init_worker():
    global _TOK
    from nanochat.tokenizer import get_tokenizer
    _TOK = get_tokenizer()


def _compute_doc(text: str):
    """Return (d1, d2, d3, d4, d5, avg_distinct, doc_tokens, doc_chars)."""
    t = text or ""
    ids = _TOK.encode(t)
    L = len(ids)
    out = np.empty(8, dtype=np.float32)
    total = 0.0
    for idx, n in enumerate((1, 2, 3, 4, 5)):
        if L < n:
            d = 1.0
        else:
            denom = L - n + 1
            uniq = len({tuple(ids[i : i + n]) for i in range(denom)})
            d = uniq / denom
        out[idx] = d
        total += d
    out[5] = total
    out[6] = float(L)
    out[7] = float(len(t))
    return out


def _process_chunk(texts):
    arr = np.empty((len(texts), 8), dtype=np.float32)
    for i, t in enumerate(texts):
        arr[i] = _compute_doc(t)
    return arr


def main():
    # Build shard_offsets list (contiguous prefix of 100 shards) using jina row counts.
    jina_offsets = json.loads((DDUDEK / "shard_offsets.json").read_text())
    shard_entries = []
    cumulative = 0
    for i in range(N_SHARDS):
        key = f"shard_{i:05d}.parquet"
        meta = jina_offsets[key]
        # jina counts are authoritative — cross-check vs base parquet on the fly later
        shard_entries.append({"shard_idx": i, "offset": cumulative, "count": int(meta["count"])})
        cumulative += int(meta["count"])
    N_TOTAL = cumulative
    print(f"[100sh] N_TOTAL = {N_TOTAL:,} docs across {N_SHARDS} shards", flush=True)
    print(f"[100sh] workers = {N_WORKERS}, chunk = {CHUNK_SIZE}", flush=True)

    # Allocate output buffers
    distinct = np.zeros((N_TOTAL, 6), dtype=np.float32)   # d1..d5, avg
    doc_tokens = np.zeros(N_TOTAL, dtype=np.int32)
    doc_chars = np.zeros(N_TOTAL, dtype=np.int32)

    t_start = time.time()
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        for entry in shard_entries:
            i = entry["shard_idx"]; offset = entry["offset"]; count = entry["count"]
            t0 = time.time()
            table = pq.read_table(POOL / f"shard_{i:05d}.parquet", columns=["text"])
            texts = table.column("text").to_pylist()
            assert len(texts) == count, (i, len(texts), count)

            chunks = [texts[j : j + CHUNK_SIZE] for j in range(0, len(texts), CHUNK_SIZE)]
            results = pool.map(_process_chunk, chunks)
            row_offset = 0
            for r in results:
                k = len(r)
                distinct[offset + row_offset : offset + row_offset + k] = r[:, :6]
                doc_tokens[offset + row_offset : offset + row_offset + k] = r[:, 6].astype(np.int32)
                doc_chars[offset + row_offset : offset + row_offset + k] = r[:, 7].astype(np.int32)
                row_offset += k
            dt = time.time() - t0
            print(
                f"  shard {i:3d}: {count:,} docs in {dt:.1f}s "
                f"({count/dt:.0f} docs/s)  avg_distinct={distinct[offset:offset+count, 5].mean():.3f} "
                f"toks_mean={doc_tokens[offset:offset+count].mean():.0f}",
                flush=True,
            )

    # Write n-gram + length arrays
    for idx, n in enumerate((1, 2, 3, 4, 5)):
        np.save(OUT / f"distinct_{n}gram_bpe.npy", distinct[:, idx])
    np.save(OUT / "avg_distinct_ngram_bpe.npy", distinct[:, 5])
    np.save(OUT / "doc_tokens.npy", doc_tokens)
    np.save(OUT / "doc_chars.npy", doc_chars)

    # Slice topic_id / format_id from jina_200shards (already covers 200 shards).
    t_full = np.load(DDUDEK / "topic_id.npy", mmap_mode="r")
    f_full = np.load(DDUDEK / "format_id.npy", mmap_mode="r")
    assert t_full.shape[0] >= N_TOTAL and f_full.shape[0] >= N_TOTAL, (t_full.shape, f_full.shape, N_TOTAL)
    topic = np.ascontiguousarray(t_full[:N_TOTAL])
    fmt = np.ascontiguousarray(f_full[:N_TOTAL])
    np.save(OUT / "topic_id.npy", topic)
    np.save(OUT / "format_id.npy", fmt)

    # Carry over name maps + write shard_offsets
    (OUT / "topic_names.json").write_text((DDUDEK / "topic_str_map.json").read_text())
    (OUT / "format_names.json").write_text((DDUDEK / "format_str_map.json").read_text())
    (OUT / "shard_offsets.json").write_text(json.dumps(shard_entries, indent=2))

    # Stats printout
    total_min = (time.time() - t_start) / 60
    print(f"\n[100sh] done in {total_min:.1f} min. wrote {OUT}/")
    print(f"  N_TOTAL = {N_TOTAL:,}")
    a = distinct[:, 5]
    print(
        f"  avg_distinct_ngram_bpe  mean={a.mean():.3f}  median={np.median(a):.3f}  "
        f"p5={np.percentile(a, 5):.3f}  p95={np.percentile(a, 95):.3f}"
    )
    print(
        f"  doc_tokens              mean={doc_tokens.mean():.0f}  median={int(np.median(doc_tokens))}  "
        f"p95={int(np.percentile(doc_tokens, 95))}"
    )
    print(f"  topic_id uniques: {len(np.unique(topic))}  format_id uniques: {len(np.unique(fmt))}")


if __name__ == "__main__":
    main()

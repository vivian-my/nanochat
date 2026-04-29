"""Data selection — ClimbMix v15.

Exploration direction: conservative stratified denoising with a tiny local
chars-per-token preference.

Idea:
- Build coarse quantile length bins.
- Within each length bin, lightly prune only the most repetitive documents
  using residualized distinct_5gram_bpe.
- Preserve the pool's natural topic×format×length-bin document-count mixture.
- Within each surviving cell, sample mostly uniformly but add a very small
  preference for documents whose chars-per-token is slightly above the cell
  median.

Atomic improvement on top of the parent:
- Make the residual distinct_5 pruning slightly more conservative in the
  longest length bins only. Prior trials suggest repetition-based pruning is
  most useful for short documents; long-form material may be over-pruned by a
  uniform lower-tail cut.

Returns `budget` unique int64 doc indices in [0, N_TOTAL).
"""
import os
import numpy as np

N_TOTAL = 8_460_288
BUDGET = int(0.838e9 / 700 * 1.4)   # 1,676,000
META = "/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_100shards"

N_TOPICS = 24
N_FORMATS = 24
N_BINS = 24


def _load(name: str) -> np.ndarray:
    return np.load(os.path.join(META, name), mmap_mode="r")


def _compute_bin_edges(tokens: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(np.asarray(tokens, dtype=np.float64), qs)
    edges = np.maximum.accumulate(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _alloc_quotas(counts: np.ndarray, budget: int) -> np.ndarray:
    total = int(counts.sum())
    if total <= 0 or budget <= 0:
        return np.zeros_like(counts, dtype=np.int64)
    target = counts.astype(np.float64) * (float(budget) / float(total))
    q = np.floor(target).astype(np.int64)
    rem = int(budget - q.sum())
    if rem > 0:
        frac = target - q
        order = np.argsort(-frac)
        q[order[:rem]] += 1
    elif rem < 0:
        frac = target - q
        order = np.argsort(frac)
        q[order[:(-rem)]] -= 1
    q = np.minimum(q, counts.astype(np.int64))
    return q


def _mix_u64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xbf58476d1ce4e5b9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return x


def _hash_uniform(idx: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(idx, dtype=np.uint64) + np.uint64(seed + 0x9E3779B97F4A7C15)
    h = _mix_u64(x)
    return ((h >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))


def _weighted_sample_without_replacement(
    rng: np.random.Generator,
    idx: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> np.ndarray:
    n = idx.shape[0]
    if k <= 0:
        return idx[:0]
    if k >= n:
        return idx.copy()
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 1e-12)
    keys = rng.random(n) ** (1.0 / w)
    part = np.argpartition(keys, n - k)[n - k:]
    return idx[part]


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    doc_tokens = _load("doc_tokens.npy")
    doc_chars = _load("doc_chars.npy")
    topic_id = _load("topic_id.npy")
    format_id = _load("format_id.npy")
    distinct5 = _load("distinct_5gram_bpe.npy")

    all_idx = np.arange(N_TOTAL, dtype=np.int64)

    # Coarse length bins for both residualization and mixture stabilization.
    edges = _compute_bin_edges(doc_tokens, n_bins=N_BINS)
    bin_id = np.digitize(doc_tokens, edges[1:-1], right=False).astype(np.int16)

    # Light pruning of repetitive tails within each length bin.
    # Atomic change: relax the pruning threshold only in the longest bins.
    keep_mask = np.ones(N_TOTAL, dtype=bool)
    long_bin_start = int(np.floor(0.75 * N_BINS))
    for b in range(N_BINS):
        idx = all_idx[bin_id == b]
        if idx.size == 0:
            continue
        d5 = np.asarray(distinct5[idx], dtype=np.float32)
        z = (d5 - d5.mean()) / (d5.std() + 1e-6)
        if idx.size < 64:
            thresh = -np.inf
        else:
            cut = 0.06 if b >= long_bin_start else 0.08
            thresh = np.quantile(z, cut)
        keep_mask[idx] = z >= thresh

    kept_idx = all_idx[keep_mask]
    kept_topic = np.asarray(topic_id[keep_mask], dtype=np.int16)
    kept_format = np.asarray(format_id[keep_mask], dtype=np.int16)
    kept_bin = np.asarray(bin_id[keep_mask], dtype=np.int16)

    # Preserve natural topic×format×length-bin composition from the full pool.
    cell_full = ((np.asarray(topic_id, dtype=np.int32) * N_FORMATS + np.asarray(format_id, dtype=np.int32)) * N_BINS
                 + np.asarray(bin_id, dtype=np.int32))
    n_cells = N_TOPICS * N_FORMATS * N_BINS
    full_counts = np.bincount(cell_full, minlength=n_cells)
    quotas = _alloc_quotas(full_counts, budget)

    # Map survivors to cells and cap quotas by available survivors.
    kept_cell = ((kept_topic.astype(np.int32) * N_FORMATS + kept_format.astype(np.int32)) * N_BINS
                 + kept_bin.astype(np.int32))
    order = np.argsort(kept_cell, kind="stable")
    kept_idx = kept_idx[order]
    kept_cell = kept_cell[order]

    uniq_cells, starts, counts = np.unique(kept_cell, return_index=True, return_counts=True)
    avail = np.zeros(n_cells, dtype=np.int64)
    avail[uniq_cells] = counts.astype(np.int64)
    quotas = np.minimum(quotas, avail)

    selected_parts = []
    leftovers = []

    # Per-cell selection: mostly uniform, tiny local cpt-above-median preference.
    for cell, start, count in zip(uniq_cells.tolist(), starts.tolist(), counts.tolist()):
        k = int(quotas[cell])
        if k <= 0:
            leftovers.append(kept_idx[start:start + count])
            continue

        idx = kept_idx[start:start + count]
        if k >= count:
            selected_parts.append(idx)
            continue

        tok = np.asarray(doc_tokens[idx], dtype=np.float32)
        ch = np.asarray(doc_chars[idx], dtype=np.float32)
        cpt = ch / np.maximum(tok, 1.0)

        med = np.median(cpt)
        mad = np.median(np.abs(cpt - med)) + 1e-6
        local = (cpt - med) / (1.4826 * mad)
        local = np.clip(local, -0.25, 1.0)
        boost = np.maximum(local, 0.0)
        weights = 1.0 + 0.08 * boost

        chosen = _weighted_sample_without_replacement(rng, idx, weights, k)
        selected_parts.append(chosen)

        chosen_sorted = np.sort(chosen)
        pos = np.searchsorted(idx, chosen_sorted)
        mask = np.ones(count, dtype=bool)
        mask[pos] = False
        leftovers.append(idx[mask])

    selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)

    # Fill any budget shortfall from surviving leftovers using near-uniform deterministic randomness.
    if selected.size < budget:
        need = int(budget - selected.size)
        if leftovers:
            pool = np.concatenate(leftovers)
        else:
            pool = kept_idx
        if pool.size > 0:
            u = _hash_uniform(pool, seed)
            take = np.argpartition(u, need - 1)[:need] if need < pool.size else np.arange(pool.size)
            fill = pool[take]
            selected = np.concatenate([selected, fill])

    # Final exact-budget / dedup safety.
    selected = np.unique(selected.astype(np.int64))
    if selected.size < budget:
        need = int(budget - selected.size)
        picked = np.zeros(N_TOTAL, dtype=bool)
        picked[selected] = True
        remaining = all_idx[~picked]
        u = _hash_uniform(remaining, seed + 17)
        take = np.argpartition(u, need - 1)[:need] if need < remaining.size else np.arange(remaining.size)
        selected = np.concatenate([selected, remaining[take]])
    elif selected.size > budget:
        u = _hash_uniform(selected, seed + 23)
        keep = np.argpartition(u, budget - 1)[:budget]
        selected = selected[keep]

    selected = np.unique(selected.astype(np.int64))
    if selected.size < budget:
        need = int(budget - selected.size)
        picked = np.zeros(N_TOTAL, dtype=bool)
        picked[selected] = True
        remaining = all_idx[~picked]
        u = _hash_uniform(remaining, seed + 29)
        take = np.argpartition(u, need - 1)[:need] if need < remaining.size else np.arange(remaining.size)
        selected = np.concatenate([selected, remaining[take]])
    elif selected.size > budget:
        u = _hash_uniform(selected, seed + 31)
        keep = np.argpartition(u, budget - 1)[:budget]
        selected = selected[keep]

    return np.sort(selected.astype(np.int64))

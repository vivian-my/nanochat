"""Data selection — ClimbMix v17 exploratory selector.

Contract: return `budget` unique int64 document ids in [0, N_TOTAL).
"""
import numpy as np

N_TOTAL = 4_485_120
BUDGET  = int(0.44e9 / 700 * 1.4)  # 879,999 with Python float rounding in the reference code
META = "/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_53shards/"


def _splitmix64(x):
    """Vectorized SplitMix64 finalizer; uint64 overflow is intentional."""
    x = np.asarray(x, dtype=np.uint64)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _hash_float(ids, seed, salt=0):
    h = _splitmix64(np.asarray(ids, dtype=np.uint64) + np.uint64(seed) * np.uint64(0xD1B54A32D192ED03) + np.uint64(salt))
    return ((h >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))


def _load(name, mmap=True):
    return np.load(META + name, mmap_mode=("r" if mmap else None))


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    """Shard-quota local declumping with weak local anomaly avoidance."""
    budget = int(budget)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if budget >= N_TOTAL:
        return np.arange(N_TOTAL, dtype=np.int64)

    # Metadata used only as weak within-neighborhood tie breakers; hash randomness dominates.
    ok = _load("ok.npy")
    topic = _load("topic_id.npy")
    fmt = _load("format_id.npy")
    tok = _load("doc_tokens.npy")
    ch = _load("doc_chars.npy")
    nf = _load("n_factual.npy")
    nr = _load("n_rerrors.npy")
    lp8 = _load("logppl_d8_ref.npy")
    lpq = _load("logppl_qwen.npy")
    div = _load("avg_distinct_ngram_bpe.npy")

    # Approximate the 53 source shards with contiguous ranges and give each an exact
    # proportional quota, avoiding a corpus-wide thinning pass.
    n_chunks = 53
    chunk_edges = np.linspace(0, N_TOTAL, n_chunks + 1, dtype=np.int64)
    lengths = chunk_edges[1:] - chunk_edges[:-1]
    ideal = lengths.astype(np.float64) * (float(budget) / float(N_TOTAL))
    quotas = np.floor(ideal).astype(np.int64)
    rem = budget - int(quotas.sum())
    if rem > 0:
        frac = ideal - quotas
        # Deterministic seed-dependent tie breaking among almost-equal remainders.
        tie = _hash_float(np.arange(n_chunks, dtype=np.uint64), seed, 0xABC98388FB8FAC03) * 1e-6
        add = np.argsort(-(frac + tie))[:rem]
        quotas[add] += 1

    selected = np.empty(budget, dtype=np.int64)
    out = 0
    last_pick = -10_000_000

    for ci in range(n_chunks):
        s = int(chunk_edges[ci]); e = int(chunk_edges[ci + 1]); q = int(quotas[ci])
        if q <= 0:
            continue
        # Exact one-per-tiny-interval lattice. Cells are mostly 5/6 docs, so this
        # is close to uniform but strongly suppresses crawl-order clumping.
        cell_edges = np.linspace(s, e, q + 1, dtype=np.int64)
        for j in range(q):
            a = int(cell_edges[j]); b = int(cell_edges[j + 1])
            if b <= a:
                continue
            ids = np.arange(a, b, dtype=np.int64)
            ids_u = ids.astype(np.uint64)

            score = _hash_float(ids_u, seed, 0x632BE59BD9B4E019)
            pen = np.zeros(ids.shape[0], dtype=np.float64)

            t = tok[ids]
            c = ch[ids]
            f = fmt[ids]
            top = topic[ids]
            p8 = lp8[ids]
            pq = lpq[ids]
            d = div[ids]

            # Rare integrity/pathology penalties. These are intentionally small:
            # the selector should not become a global quality filter.
            pen += (~ok[ids]).astype(np.float64) * 1.25
            pen += (t < 64).astype(np.float64) * 0.22
            pen += (t < 32).astype(np.float64) * 0.45
            pen += (c >= 7900).astype(np.float64) * 0.15
            ratio = c.astype(np.float64) / np.maximum(t.astype(np.float64), 1.0)
            pen += ((ratio < 2.15) | (ratio > 8.7)).astype(np.float64) * 0.10
            pen += (f == 6).astype(np.float64) * 0.18       # truncated
            pen += (f == 18).astype(np.float64) * 0.10      # spam/ads
            pen += (top == 0).astype(np.float64) * 0.08     # adult
            pen += (nf[ids] >= 4).astype(np.float64) * 0.11
            pen += (nr[ids] >= 3).astype(np.float64) * 0.10
            pen += ((p8 > 7.7) | (p8 < 1.55)).astype(np.float64) * 0.12
            pen += ((pq > 4.85) | (pq < 1.20)).astype(np.float64) * 0.08
            pen += ((d < 3.05) | (d > 4.93)).astype(np.float64) * 0.10

            # New exploratory signal: local contrast against the immediate tiny
            # crawl neighborhood. If one candidate is a length/PPL/diversity
            # outlier relative to its 5/6-doc cell, mildly avoid it.
            if ids.size >= 3:
                lt = np.log1p(t.astype(np.float64))
                med_lt = np.median(lt)
                med_p8 = np.median(p8.astype(np.float64))
                med_d = np.median(d.astype(np.float64))
                pen += (np.abs(lt - med_lt) > 0.95).astype(np.float64) * 0.055
                pen += (np.abs(p8.astype(np.float64) - med_p8) > 1.35).astype(np.float64) * 0.050
                pen += (np.abs(d.astype(np.float64) - med_d) > 0.48).astype(np.float64) * 0.045

            # Immediate-neighbor near-duplicate/template proxy in original order.
            # Penalize matching topic/format with very similar size and PPL to
            # adjacent documents, but only weakly so that natural runs remain.
            sim = np.zeros(ids.shape[0], dtype=bool)
            valid_prev = ids > 0
            if np.any(valid_prev):
                ii = ids[valid_prev]
                close_prev = (
                    (topic[ii - 1] == topic[ii]) & (fmt[ii - 1] == fmt[ii]) &
                    (np.abs(tok[ii - 1].astype(np.int64) - tok[ii].astype(np.int64)) <= 5) &
                    (np.abs(ch[ii - 1].astype(np.int64) - ch[ii].astype(np.int64)) <= 45) &
                    (np.abs(lp8[ii - 1] - lp8[ii]) <= 0.08)
                )
                sim[np.nonzero(valid_prev)[0]] |= close_prev
            valid_next = ids < (N_TOTAL - 1)
            if np.any(valid_next):
                ii = ids[valid_next]
                close_next = (
                    (topic[ii + 1] == topic[ii]) & (fmt[ii + 1] == fmt[ii]) &
                    (np.abs(tok[ii + 1].astype(np.int64) - tok[ii].astype(np.int64)) <= 5) &
                    (np.abs(ch[ii + 1].astype(np.int64) - ch[ii].astype(np.int64)) <= 45) &
                    (np.abs(lp8[ii + 1] - lp8[ii]) <= 0.08)
                )
                sim[np.nonzero(valid_next)[0]] |= close_next
            pen += sim.astype(np.float64) * 0.075

            # Greedy cross-cell hard-core nudge: avoid the only remaining way to
            # create adjacent selections under the one-per-interval lattice.
            if last_pick >= 0:
                pen += (ids == last_pick + 1).astype(np.float64) * 0.38
                pen += (ids == last_pick + 2).astype(np.float64) * 0.055

            sc = score + pen
            pick = int(ids[int(np.argmin(sc))])
            selected[out] = pick
            out += 1
            last_pick = pick

    # Robustness for unusual budget values or any empty-cell corner case.
    if out != budget:
        selected = selected[:out]
        mask = np.zeros(N_TOTAL, dtype=bool)
        mask[selected] = True
        if out < budget:
            all_ids = np.arange(N_TOTAL, dtype=np.int64)
            avail = all_ids[~mask]
            need = budget - out
            h = _hash_float(avail.astype(np.uint64), seed, 0x9E6C63D0676A9A99)
            fill = avail[np.argpartition(h, need - 1)[:need]]
            selected = np.concatenate([selected, fill.astype(np.int64)])
        else:
            h = _hash_float(selected.astype(np.uint64), seed, 0xD6E8FEB86659FD93)
            keep = np.argpartition(h, budget - 1)[:budget]
            selected = selected[keep]

    return np.sort(selected.astype(np.int64, copy=False))

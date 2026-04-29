"""Data selection — ClimbMix v18 exploratory natural-band selector.

This trial deliberately avoids another broad soft ranker.  It builds a large
"natural band" of documents that are not extremely short/long, not reference-LM
outliers, and not heavily marked by the Gemini annotation counts, then samples
mostly uniformly from that band while using a small deterministic reserve to keep
format/topic coverage from collapsing.
"""
import numpy as np

N_TOTAL = 4_485_120
BUDGET = 880_000

BASE: dict = {}
COMPOSITES: dict = {}
LEXICALS: dict = {}


def feature_specs():
    """Return composite feature definitions and no lexical features for this trial."""

    def anno_load_v1(B):
        rs = B["n_rsteps"].astype(np.float32)
        re = B["n_rerrors"].astype(np.float32)
        nf = B["n_factual"].astype(np.float32)
        bad = (rs < 0) | (re < 0) | (nf < 0)
        # High rsteps looked harmful in the provided ablation; factual/reasoning
        # errors are treated as stronger signs of noisy/generated pages.
        out = 0.45 * np.maximum(rs, 0.0) + 1.6 * np.maximum(re, 0.0) + 1.15 * np.maximum(nf, 0.0)
        out[bad] = 9.0
        return out.astype(np.float32)

    def model_outlier_v1(B):
        t = np.maximum(B["doc_tokens"].astype(np.float32), 1.0)
        lp = B["logppl_d8_ref"].astype(np.float32)
        lq = B["logppl_qwen"].astype(np.float32)
        div = B["avg_distinct_ngram_bpe"].astype(np.float32)
        x = np.log1p(t)
        # simple length residuals, constants chosen from pool-level moments in prompt
        rd = lp - (7.55 - 0.63 * x)
        rq = lq - (4.85 - 0.34 * x)
        rdiv = div - (5.70 - 0.23 * x)
        # Outlier magnitude rather than monotone quality preference.
        z = (np.abs(rd) / 1.05) + 0.55 * (np.abs(rq) / 0.62) + 0.65 * (np.abs(rdiv) / 0.30)
        z += 0.25 * np.maximum(0.0, 2.0 - lp) + 0.18 * np.maximum(0.0, lp - 6.2)
        return z.astype(np.float32)

    def token_band_v1(B):
        t = B["doc_tokens"].astype(np.float32)
        c = B.get("doc_chars", t * 4.0).astype(np.float32)
        cpt = c / np.maximum(t, 1.0)
        # Penalty for the very short pages that can waste training steps and for
        # unusually long/truncated pages; also catch odd char/token ratios.
        p = np.zeros_like(t, dtype=np.float32)
        p += np.maximum(0.0, (180.0 - t) / 110.0)
        p += 0.55 * np.maximum(0.0, (t - 1550.0) / 550.0)
        p += 0.45 * np.maximum(0.0, 2.7 - cpt)
        p += 0.25 * np.maximum(0.0, cpt - 6.8)
        return p.astype(np.float32)

    return [("anno_load_v1", anno_load_v1), ("model_outlier_v1", model_outlier_v1), ("token_band_v1", token_band_v1)], []


def _largest_remainder_counts(keys: np.ndarray, eligible: np.ndarray, budget: int):
    """Proportional integer quotas over key values restricted to eligible docs."""
    vals, cnt = np.unique(keys[eligible], return_counts=True)
    raw = cnt.astype(np.float64) * (float(budget) / float(cnt.sum()))
    q = np.floor(raw).astype(np.int64)
    rem = int(budget - q.sum())
    if rem > 0:
        order = np.argsort(-(raw - q), kind="mergesort")[:rem]
        q[order] += 1
    return vals.astype(np.int64), q.astype(np.int64)


def _weighted_take(rng, inds: np.ndarray, k: int, weight: np.ndarray) -> np.ndarray:
    if k <= 0 or inds.size == 0:
        return np.empty(0, dtype=np.int64)
    if inds.size <= k:
        return inds.astype(np.int64, copy=False)
    w = weight[inds].astype(np.float64, copy=False)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1e-6)
    # Efraimidis-Spirakis priority sampling; avoids normalizing probabilities for
    # thousands of small cells and is stable for capped weights.
    u = rng.random(inds.size)
    key = -np.log(np.maximum(u, 1e-12)) / w
    part = np.argpartition(key, k - 1)[:k]
    return inds[part].astype(np.int64, copy=False)


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    ok = BASE.get("ok", np.ones(N_TOTAL, dtype=bool)).astype(bool)
    tok = BASE["doc_tokens"].astype(np.float32)
    lp = BASE["logppl_d8_ref"].astype(np.float32)
    lq = BASE["logppl_qwen"].astype(np.float32)
    div = BASE["avg_distinct_ngram_bpe"].astype(np.float32)
    rs = BASE["n_rsteps"]
    re = BASE["n_rerrors"]
    nf = BASE["n_factual"]

    anno = COMPOSITES.get("anno_load_v1")
    mout = COMPOSITES.get("model_outlier_v1")
    tpen = COMPOSITES.get("token_band_v1")
    if anno is None:
        anno = np.zeros(N_TOTAL, dtype=np.float32)
    if mout is None:
        mout = np.zeros(N_TOTAL, dtype=np.float32)
    if tpen is None:
        tpen = np.zeros(N_TOTAL, dtype=np.float32)

    # Hard natural-band gate.  This intentionally tests whether removing the
    # noisiest/most synthetic-looking tail is better than the mostly-soft recipes
    # tried so far.  Thresholds are loose enough to retain several times budget.
    candidate = (
        ok
        & (tok >= 120.0) & (tok <= 1850.0)
        & np.isfinite(lp) & np.isfinite(lq) & np.isfinite(div)
        & (lp >= 2.05) & (lp <= 6.35)
        & (lq >= 1.55) & (lq <= 4.35)
        & (div >= 3.35) & (div <= 4.88)
        & (rs >= 0) & (re >= 0) & (nf >= 0)
        & (rs <= 4) & (re <= 1) & (nf <= 2)
        & (anno <= 4.6) & (mout <= 3.9) & (tpen <= 1.6)
    )

    # If the gate is unexpectedly too small under future metadata changes, relax
    # gracefully rather than violating the contract.
    if int(candidate.sum()) < budget:
        candidate = (
            ok
            & (tok >= 90.0) & (tok <= 2100.0)
            & np.isfinite(lp) & np.isfinite(lq) & np.isfinite(div)
            & (lp >= 1.85) & (lp <= 7.10)
            & (lq >= 1.35) & (lq <= 4.85)
            & (div >= 3.05) & (div <= 4.95)
            & (rs >= 0) & (re >= 0) & (nf >= 0)
            & (rs <= 6) & (re <= 2) & (nf <= 3)
        )
    if int(candidate.sum()) < budget:
        candidate = ok & (tok >= 50.0) & np.isfinite(lp) & np.isfinite(lq) & np.isfinite(div)

    # Preserve broad topic-format proportions inside the cleaned band.  To avoid
    # brittle empty tiny cells, quota on topic*24+format, proportional to the band
    # itself rather than the whole pool.
    topic = BASE["topic_id"].astype(np.int32)
    fmt = BASE["format_id"].astype(np.int32)
    key = (topic * 24 + fmt).astype(np.int32)

    # Within the band, sampling remains close to uniform; a capped weight only
    # de-emphasizes the residual worst tail and mildly favors enough tokens.
    risk = 0.72 * anno.astype(np.float32) + 0.58 * mout.astype(np.float32) + 0.85 * tpen.astype(np.float32)
    length_boost = np.clip(np.sqrt(np.maximum(tok, 1.0) / 520.0), 0.72, 1.32)
    weight = length_boost * np.exp(-0.20 * np.clip(risk, 0.0, 8.0))
    weight = np.clip(weight, 0.20, 1.55).astype(np.float32)

    vals, quotas = _largest_remainder_counts(key, candidate, budget)
    selected_parts = []
    total = 0
    for v, q in zip(vals, quotas):
        if q <= 0:
            continue
        inds = np.flatnonzero(candidate & (key == v))
        take = _weighted_take(rng, inds, int(q), weight)
        selected_parts.append(take)
        total += take.size

    if selected_parts:
        selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    else:
        selected = np.empty(0, dtype=np.int64)

    # Repair any shortfall/overshoot from rare-cell edge cases.
    if selected.size > budget:
        keep = rng.choice(selected.size, size=budget, replace=False)
        selected = selected[keep]
    elif selected.size < budget:
        chosen = np.zeros(N_TOTAL, dtype=bool)
        chosen[selected] = True
        rem_pool = np.flatnonzero(ok & (~chosen))
        need = budget - selected.size
        fill = rng.choice(rem_pool, size=need, replace=False).astype(np.int64)
        selected = np.concatenate([selected, fill])

    # Final contract cleanup; normally no-op.
    if selected.size != budget or np.unique(selected).size != selected.size:
        selected = np.unique(selected.astype(np.int64, copy=False))
        if selected.size < budget:
            chosen = np.zeros(N_TOTAL, dtype=bool)
            chosen[selected] = True
            pool = np.flatnonzero(ok & (~chosen))
            fill = rng.choice(pool, size=budget - selected.size, replace=False).astype(np.int64)
            selected = np.concatenate([selected, fill])
        elif selected.size > budget:
            selected = selected[rng.choice(selected.size, size=budget, replace=False)]

    return np.sort(selected.astype(np.int64, copy=False))

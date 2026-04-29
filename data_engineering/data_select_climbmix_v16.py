"""Data selection — ClimbMix v16: intersected log-token and char-length middle-band random sampling.

Returns `budget` unique int64 doc indices in [0, N_TOTAL).

Idea explored:
- Keep the strong near-random character of the parent selector.
- Trim multiplicative extremes in log-token space as before.
- Add one atomic change: also require documents to lie in a broad middle band
  of character length, then sample uniformly at random from the intersection.

This tests whether token-length and character-length each catch slightly
different pathological extremes (very short/truncated pages, markup-heavy or
unusually verbose outliers) while preserving broad coverage.
"""
import os

import numpy as np

N_TOTAL = 8_460_288
BUDGET = int(0.44e9 / 700 * 1.4)   # 880,000
META = "/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_100shards"


def _load_tokens() -> np.ndarray:
    return np.load(os.path.join(META, "doc_tokens.npy"), mmap_mode="r")


def _load_chars() -> np.ndarray:
    return np.load(os.path.join(META, "doc_chars.npy"), mmap_mode="r")


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tokens = _load_tokens()
    chars = _load_chars()

    # Parent's successful filter: conservative middle-band in log-token space.
    log_tokens = np.log(np.maximum(tokens.astype(np.float64), 1.0))
    tok_low_q, tok_high_q = np.quantile(log_tokens, [0.08, 0.92])

    # Single added change: require a broad middle band in character length too.
    char_low_q, char_high_q = np.quantile(chars, [0.05, 0.95])

    eligible = np.flatnonzero(
        (log_tokens >= tok_low_q)
        & (log_tokens <= tok_high_q)
        & (chars >= char_low_q)
        & (chars <= char_high_q)
    )

    if eligible.size < budget:
        # Safety backfill: relax to the parent token-only filter if the
        # intersection is unexpectedly too small.
        eligible = np.flatnonzero((log_tokens >= tok_low_q) & (log_tokens <= tok_high_q))

    if eligible.size < budget:
        # Final safety fallback.
        idx = rng.choice(N_TOTAL, size=budget, replace=False)
        return np.sort(idx.astype(np.int64))

    chosen = rng.choice(eligible, size=budget, replace=False)
    return np.sort(chosen.astype(np.int64))

"""Data selection — ClimbMix v7 (full 9-feature bank, ablation-prior-equipped).

Same pool, budget, and training config as v5/v6 (scores directly comparable).

Pool: 4,485,120 docs across 53 shards (shards 0..52 of base_data_climbmix_full).
Features (aligned by global doc id g, all under the same meta dir):

    meta = '/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_53shards/'

    # --- annotation (Gemini rubric; int16, -1 if annotation failed)
    n_factual  = np.load(meta+'n_factual.npy')       int16    factual errors
    n_rsteps   = np.load(meta+'n_rsteps.npy')        int16    reasoning steps
    n_rerrors  = np.load(meta+'n_rerrors.npy')       int16    invalid reasoning steps (⊆ n_rsteps)

    # --- classifier (WebOrganizer; int32, 0..23)
    topic_id   = np.load(meta+'topic_id.npy')        int32    24 topic classes
    format_id  = np.load(meta+'format_id.npy')       int32    24 format classes

    # --- reference-model perplexity
    logppl_qwen   = np.load(meta+'logppl_qwen.npy')   float32  Qwen-2.5-0.5B mean NLL
    logppl_d8_ref = np.load(meta+'logppl_d8_ref.npy') float32  d8-ppl-ref mean NLL
    ppl_qwen      = np.load(meta+'ppl_qwen.npy')      float32  = exp(logppl_qwen)   [heavy-tailed]
    ppl_d8_ref    = np.load(meta+'ppl_d8_ref.npy')    float32  = exp(logppl_d8_ref) [heavy-tailed]

    # --- n-gram diversity (nanochat BPE)
    avg_distinct     = np.load(meta+'avg_distinct_ngram_bpe.npy')  float32  ∑_{n=1..5} distinct_n   (range ≈ [0, 5])
    distinct_1gram   = np.load(meta+'distinct_1gram_bpe.npy')      float32
    distinct_2gram   = np.load(meta+'distinct_2gram_bpe.npy')      float32
    distinct_3gram   = np.load(meta+'distinct_3gram_bpe.npy')      float32
    distinct_4gram   = np.load(meta+'distinct_4gram_bpe.npy')      float32
    distinct_5gram   = np.load(meta+'distinct_5gram_bpe.npy')      float32

    # --- doc length
    doc_tokens = np.load(meta+'doc_tokens.npy')      int32    nanochat-BPE length
    doc_chars  = np.load(meta+'doc_chars.npy')       int32    char length (≤ 8000)

    # --- mask
    ok         = np.load(meta+'ok.npy')              bool     True iff annotation succeeded

Return `budget` unique int64 doc indices in [0, N_TOTAL).
BUDGET and N_TOTAL are fixed — do not change them.
"""
import numpy as np

N_TOTAL = 4_485_120
BUDGET  = int(0.44e9 / 700 * 1.4)  # 879,999 — same d8 token budget as v4/v5/v6


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    """Random-uniform baseline. Replace with a smarter recipe."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(N_TOTAL, size=budget, replace=False)
    return np.sort(idx.astype(np.int64))

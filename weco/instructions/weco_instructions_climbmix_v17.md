# WECO ClimbMix v17 — data-selection instructions (full 9 features, ablation priors)

## Goal

Write `select_docs(budget, seed)` to pick **880,000** documents from a
**4,485,120-doc pool** that minimize the mean `val_bpb` of a d8 model trained
on the selected set.

The trainer averages **3 independent seeds** (~440 M tokens each, d8 FP8,
target_param_ratio = 10.5). Report both mean and per-seed std — selections
that beat random on mean but inflate variance are less useful than tight ones.

**Reference**:

| selection | val_bpb |
|---|---|
| `random_uniform` (4-seed) | 0.951452 ± 0.000529 |

Targets: beat random by **>0.001** (≥ 1.9σ) to be clearly useful. Improvements
< 5e-4 are within the noise floor.

Budget: 880,000 / 4,485,120 = **19.6 % keep rate**.

---

## Single-feature ablation grid (priors)

Each row is a 3-seed d8 trained on a single-feature filter applied to the
same pool, with the same training config as this run. These are the
**hard-rule filter outcomes** for the 9 features available.

| trial | rule | val_bpb (3-seed) | Δ vs random | sigma |
|---|---|---:|---:|---:|
| `ppl_d8_highest` | TOP 19.6 % `logppl_d8_ref` | 1.0252 | +0.074 | **−139σ** |
| `ngram_top` | TOP 19.6 % `avg_distinct` | 1.0144 | +0.063 | −119σ |
| `ppl_qwen_highest` | TOP 19.6 % `logppl_qwen` | 1.0068 | +0.055 | −105σ |
| `ppl_d8_lowest` | BOT 19.6 % `logppl_d8_ref` | 0.9863 | +0.035 | −66σ |
| `ppl_qwen_lowest` | BOT 19.6 % `logppl_qwen` | 0.9848 | +0.033 | −63σ |
| `ngram_bot` | BOT 19.6 % `avg_distinct` | 0.9676 | +0.016 | −30σ |
| `length_long` | TOP 19.6 % `doc_tokens` | 0.9605 | +0.009 | −17σ |
| `central_len` | central 19.6 % `doc_tokens` | 0.9619 | +0.010 | −20σ |
| `length_short` | BOT 19.6 % `doc_tokens` (shortest) | **3.51** | +2.55 | training collapse — too few tokens |
| `rstep_gt1` | `n_rsteps > 1` + random fill | 0.9537 | +0.0022 | −4.2σ |
| `fact_zero` | `n_factual == 0` + random fill | 0.9528 | +0.0013 | −2.5σ |
| `rerr_zero` | `n_rerrors == 0` + random fill | 0.9517 | +0.0002 | **−0.4σ (≈ random)** |
| `topic_quota` | per-topic proportional, random within | 0.9516 | +0.0001 | **−0.2σ (≈ random)** |
| **`random_uniform`** | (baseline) | **0.9515** | **0** | **0** |

---

## The pool — ClimbMix (pre-curated)

Pre-filtered by quality classification, dedup, diversity balancing.
Pool distribution (ok docs only, first 53 shards):

- **Topics** — 24 classes. Science & Tech 24 %, Health 15 %, Home & Hobbies 12 %.
- **Formats** — 24 classes. Knowledge Article 18 %, Tutorial 15 %, Product Page 9 %.
- 4,483,659 ok / 4,485,120 total (1,461 docs have −1 annotations).

---

## Feature bank (9 features)

All arrays at `/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_53shards/`,
indexed by global doc id `g ∈ [0, 4_485_120)`. Read-only.

| # | File | dtype | Description |
|---|---|---|---|
| 1 | `topic_id.npy` | int32 | WebOrganizer topic id, 0..23 |
| 2 | `format_id.npy` | int32 | WebOrganizer format id, 0..23 |
| 3 | `n_rsteps.npy` | int16 | reasoning steps |
| 4 | `n_rerrors.npy` | int16 | invalid reasoning steps (⊆ `n_rsteps`) |
| 5 | `n_factual.npy` | int16 | factual errors |
| 6 | `logppl_qwen.npy` | float32 | Qwen-2.5-0.5B mean per-token NLL |
| 7 | `logppl_d8_ref.npy` | float32 | d8 (held-out shards 60..559) mean per-token NLL |
| 8 | `avg_distinct_ngram_bpe.npy` | float32 | Σ_{n=1..5} distinct_n (range ≈ [0, 5]) |
| 9 | `doc_tokens.npy` | int32 | nanochat-BPE doc length |

**Utility columns**: `ok.npy` (bool), `doc_chars.npy` (int32),
`distinct_Ngram_bpe.npy` per N=1..5, `topic_names.json`, `format_names.json`.

### Feature definitions

#### (1) topic_id — WebOrganizer topic classifier

| id | name | id | name | id | name |
|---:|---|---:|---|---:|---|
| 0 | Adult | 1 | Art & Design | 2 | Software Dev. |
| 3 | Crime & Law | 4 | Education & Jobs | 5 | Hardware |
| 6 | Entertainment | 7 | Social Life | 8 | Fashion & Beauty |
| 9 | Finance & Business | 10 | Food & Dining | 11 | Games |
| 12 | Health | 13 | History | 14 | Home & Hobbies |
| 15 | Industrial | 16 | Literature | 17 | Politics |
| 18 | Religion | 19 | Science & Tech. | 20 | Software |
| 21 | Sports & Fitness | 22 | Transportation | 23 | Travel |

#### (2) format_id — WebOrganizer format classifier

| id | name | id | name | id | name |
|---:|---|---:|---|---:|---|
| 0 | Academic Writing | 1 | Content Listing | 2 | Creative Writing |
| 3 | Customer Support | 4 | Comment Section | 5 | FAQ |
| 6 | Truncated | 7 | Knowledge Article | 8 | Legal Notices |
| 9 | Listicle | 10 | News Article | 11 | Nonfiction Writing |
| 12 | About (Org.) | 13 | News (Org.) | 14 | About (Pers.) |
| 15 | Personal Blog | 16 | Product Page | 17 | Q&A Forum |
| 18 | Spam / Ads | 19 | Structured Data | 20 | Documentation |
| 21 | Audio Transcript | 22 | Tutorial | 23 | User Review |

#### (3) n_rsteps — reasoning steps
Inferential moves counted by Gemini rubric. Pool: μ=1.81, p95=4, max=36, zero-rate 4 %.

#### (4) n_rerrors — invalid reasoning steps
By construction `n_rerrors ≤ n_rsteps`. Pool: μ=0.27, p95=1, max=20, zero-rate 77 %.

#### (5) n_factual — factual errors
Verifiable mistakes (dates, entities, scientific facts). Pool: μ=0.35, p95=2, max=40, zero-rate 72 %.

#### (6) logppl_qwen — out-of-distribution LM
Per-doc mean NLL from Qwen-2.5-0.5B base. Qwen BPE, 2048-token truncation. **Prefer log form over `ppl_qwen`** (= `exp(logppl)`, heavy-tailed). Pool: μ=2.732, σ=0.663, p[1,25,50,75,95,99]=[1.29, 2.35, 2.70, 3.08, 3.80, 4.61].

#### (7) logppl_d8_ref — in-distribution LM
Per-doc mean NLL from a nanochat d8 FP8 trained on ClimbMix shards 60..559 (held out from selection pool). nanochat BPE, aligned with `doc_tokens`. Pool: μ=3.674, σ=1.122, p[1,25,50,75,95,99]=[1.75, 3.07, 3.52, 4.06, 5.54, 7.46].

#### (8) avg_distinct_ngram_bpe — diversity
`Σ_{n=1..5} (distinct n-grams / total n-grams)` in nanochat BPE. SUM, not mean. Range ≈ [0, 5]. Pool: μ=4.258, σ=0.327, p[1,25,50,75,95,99]=[3.18, 4.10, 4.28, 4.46, 4.73, 4.89].

#### (9) doc_tokens — nanochat-BPE length
Pool: mean 545, p[1,50,95,99]=[42, 527, 1447, 1807]. Pool total tokens ≈ 2.44 B for ok docs.

---

## Known correlations (Spearman ρ on 500K-doc sample)

| pair | ρ |
|---|---:|
| `logppl_qwen` ↔ `logppl_d8_ref` | +0.88 |
| `avg_distinct` ↔ `distinct_1g/2g/3g` | +0.90/0.98/0.92 |
| `doc_tokens` ↔ `doc_chars` | +0.98 |
| `logppl_d8_ref` ↔ `doc_tokens` | **−0.60** (length confound) |
| `avg_distinct` ↔ `doc_tokens` | **−0.66** (length confound) |
| `logppl_d8_ref` ↔ `avg_distinct` | +0.67 |
| `n_rsteps` ↔ `doc_tokens` | +0.48 |
| `n_rerrors` ↔ `n_factual` | +0.21 |
| `n_rsteps` ↔ `n_rerrors` | +0.07 |
| `n_rsteps` ↔ `n_factual` | −0.01 |

Effectively-independent continuous signals:
1 PPL · 1 diversity · `doc_tokens` · `n_rsteps` · `n_rerrors` · `n_factual`.
Plus `topic_id`, `format_id` as categorical moderators.

---

## Contract

```python
def select_docs(budget: int, seed: int) -> np.ndarray:
    # returns exactly `budget` unique int64 indices in [0, 4_485_120)
```

BUDGET is fixed at 880,000. Any other shape, duplicates, or out-of-range
indices → `val_bpb = 9.9999` (penalty).

**Constraints:**
- Read only from the `meta_53shards/` directory above.
- No O(N²) ops over 4.5 M docs.
- No external network, no downloads.
- numpy + stdlib only.

---

## Feedback after each trial

| metric | meaning |
|---|---|
| `val_bpb` | mean over 3 seeds (primary signal, lower = better) |
| `val_bpb_std` | 3-seed std (want this small) |
| per-seed bpb | for variance diagnosis |
| `n_docs` | should equal 880,000 |

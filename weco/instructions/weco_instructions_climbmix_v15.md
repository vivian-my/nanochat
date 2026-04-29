# WECO ClimbMix v15 — data-selection instructions (cheap features, 100-shard pool)

## Goal

Write `select_docs(budget, seed)` to pick **1,676,000** documents from an
**8,460,288-doc pool** that minimize the mean `val_bpb` of a d8 model trained
on the selected set.

The trainer averages **4 independent seeds** (~838 M tokens each, d8 FP8,
target_param_ratio = 20). Report both mean and per-seed std — selections that
beat random on mean but inflate variance are less useful than tight ones.

Budget: 1,676,000 / 8,460,288 = **19.8 % keep rate**.

Targets: improvements on the order of **~1e-3** (a few × the 4-seed std) are
clearly useful; <5e-4 is within the noise floor.

---

## The pool — ClimbMix (pre-curated)

Pre-filtered by quality classification, dedup, and diversity balancing.
Pool: 8,460,288 docs across 100 shards (shards 0..99 of the climbmix base).

- **Topics** — 24 classes (WebOrganizer). Top: Science & Tech, Health, Home & Hobbies.
- **Formats** — 24 classes (WebOrganizer). Top: Knowledge Article, Tutorial, Product Page.

---

## Feature bank (cheap features only)

All arrays at `/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_100shards/`,
indexed by global doc id `g ∈ [0, 8_460_288)`. Read-only.

| # | File | dtype | Description |
|---|---|---|---|
| 1 | `topic_id.npy` | int32 | WebOrganizer topic id, 0..23 |
| 2 | `format_id.npy` | int32 | WebOrganizer format id, 0..23 |
| 3 | `avg_distinct_ngram_bpe.npy` | float32 | Σ_{n=1..5} distinct_n (range ≈ [0, 5]) |
| 4 | `doc_tokens.npy` | int32 | nanochat-BPE doc length (full doc) |

**Utility / per-n columns**: `doc_chars.npy` (int32, full-doc characters),
`distinct_{1,2,3,4,5}gram_bpe.npy` (float32, per-n components of avg_distinct),
`topic_names.json`, `format_names.json`, `shard_offsets.json`.

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

#### (3) avg_distinct_ngram_bpe — n-gram diversity

For each doc and each `n ∈ {1,2,3,4,5}`:

```
distinct_n = (# unique n-grams) / (# total n-grams)     # nanochat BPE tokens
```

Then `avg_distinct_ngram_bpe = distinct_1 + distinct_2 + distinct_3 + distinct_4 + distinct_5`
(SUM across n, not mean). Range ≈ [0, 5]. Higher = every short span unique
(diverse prose). Lower = repetitive, templated, SEO-farm, tabular.

Pool: μ=4.258, σ=0.327, p[1,25,50,75,95,99] = [3.18, 4.10, 4.28, 4.46, 4.73, 4.89].
Per-n means: d1=0.53, d2=0.86, d3=0.93, d4=0.96, d5=0.97.

#### (4) doc_tokens — nanochat-BPE doc length

Integer length of the full document in nanochat BPE tokens.

Pool: mean 636, p[1,50,95,99] = [42, 553, 1539, 1894]. Pool total tokens ≈ 5.38 B.

`doc_chars.npy` is the full character length (Spearman ρ vs `doc_tokens` ≈ +0.98).

---

## Known correlations (Spearman ρ on pool sample)

| pair | ρ |
|---|---:|
| `avg_distinct` ↔ `distinct_1g/2g/3g` | +0.90 / +0.98 / +0.92 |
| `doc_tokens` ↔ `doc_chars` | +0.98 |
| `avg_distinct` ↔ `doc_tokens` | **−0.66** (length confound — Heaps' law) |
| `distinct_1gram` ↔ `doc_tokens` | **−0.83** |
| `distinct_5gram` ↔ `doc_tokens` | −0.38 |

`avg_distinct` is heavily length-confounded (TTR shrinks mechanically with
length). A raw n-gram filter is also a covert length filter. Residualize
against `doc_tokens` (e.g., z-score within length decile) if you want a
cleaner content signal. Per-n distinct rates are less length-confounded for
larger n (`distinct_5gram` is the cleanest).

---

## Contract

```python
def select_docs(budget: int, seed: int) -> np.ndarray:
    # returns exactly `budget` unique int64 indices in [0, 8_460_288)
```

BUDGET is fixed at 1,676,000. Any other shape, duplicates, or out-of-range
indices → `val_bpb = 9.9999` (penalty).

**Constraints:**
- Read only from the `meta_100shards/` directory above.
- No O(N²) ops over 8.5 M docs.
- No external network, no downloads.
- numpy + stdlib only.

---

## Feedback after each trial

| metric | meaning |
|---|---|
| `val_bpb` | mean over 4 seeds (primary signal, lower = better) |
| `val_bpb_std` | 4-seed std (want this small) |
| per-seed bpb | for variance diagnosis |
| `n_docs` | should equal 1,676,000 |

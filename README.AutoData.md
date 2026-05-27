# AutoData — LLM-driven Data Selection for nanochat

This fork extends [Karpathy's nanochat](https://github.com/karpathy/nanochat) with
**AutoData**, a system that uses a frontier LLM (GPT-5.5 / Claude-Opus-4.7 /
Gemini-3-Pro) plus the [WECO](https://weco.ai) optimisation loop to *discover*
data-selection recipes for the ClimbMix pre-training corpus. The discovered
recipes select subsets of the 553 M-document ClimbMix pool that train smaller
nanochat models to higher CORE / lower val_bpb than human-designed baselines
(Random, RegMix, DCLM, PPL-filter).

The upstream `README.md` is unchanged. This file describes the AutoData
additions; the two end-to-end guides under [`docs/`](docs/) cover the full
pipeline.

## Pipeline at a glance

```
                                          ┌─────────────────────┐
   features                               │  WECO Optimisation  │
  (length, ngram,                         │  Loop (200 steps)   │
   ppl, topic, format)  ──── inputs ──▶   │                     │
                                          │   LLM generates     │
                                          │   select_docs() ─┐  │
                                          │   in Python      │  │
                                          │                  │  │
                                          │   eval CORE  ◀───┘  │
                                          │   on d8 r=10        │
                                          └─────────────────────┘
                                                    │
                                          best step │ selector
                                                    ▼
                                          ┌─────────────────────┐
                                          │ Train nanochat at   │
                                          │ d8 / d12 / d24      │
                                          │ (Modal H100:8 DDP)  │
                                          └─────────────────────┘
                                                    │
                                                    ▼
                                          val_bpb + CORE + 22-task
                                          breakdown logged to JSON
```

## Two-step usage

| Step | What | Guide |
|---|---|---|
| **1. Search** | Run a WECO optimisation loop — the LLM proposes 200 candidate `select_docs()` functions; each is evaluated by training nanochat-d8 (~10 min on 1 H100) and scoring on val_bpb *or* CORE. | [`docs/01-weco-search.md`](docs/01-weco-search.md) |
| **2. Train at scale** | Take the best step's selector, materialise the 14.37 M-doc selection into parquets, and train nanochat at d12 / d16 / d20 / d24 on Modal H100:8 DDP. | [`docs/02-train-from-recipe.md`](docs/02-train-from-recipe.md) |

A complete handoff doc with current results, key file paths, and lessons
learned lives at [`summary.md`](summary.md).

## Headline results (5-seed trimmed, d24 r=8)

| Method | val_bpb ↓ | CORE ↑ |
|---|---|---|
| Random uniform | 0.7061 ± 0.0004 | 0.2579 |
| DCLM-Baseline | 0.7499 ± 0.0001 | 0.2470 |
| PPL filter | 0.7267 ± 0.0001 | 0.2535 |
| RegMix-topic24 | 0.7124 ± 0.0002 | 0.2531 |
| NanoChat default (Sequential) | 0.7065 ± 0.0004 | 0.2667 |
| **AutoData (val-bpb)** GPT-5.5 s149 | 0.7066 ± 0.0003 | 0.2596 |
| **AutoData (CORE)** GPT-5.5 step-105 | **0.7066 ± 0.0001** | **0.2695** |

The full per-scale (d8/d12/d16/d20/d24) and per-task (22 CORE tasks)
breakdowns live in [`eval/all_results.json`](eval/all_results.json) and
[`eval/summary.md`](eval/summary.md). The mirror on Modal is at
`nanochat-climbmix-scratch:eval/all_results.json`.

## What's in this fork

| Path | Purpose |
|---|---|
| `experiments/climbmix_full_v1/` | WECO selector source files (`data_select_*.py`), eval harnesses (`weco_eval_full*.py`), launch scripts (`launch/*.sh`), and LLM instructions (`instructions/*.md`). |
| `experiments/main_baselines/` | Recipe-to-Modal pipeline: build parquet sets from a selector index, upload to scratch volume, spawn `train_row` H100:8 trainings, collect results. |
| `data_engineering/modal_baselines/modal_main_baselines.py` | The Modal app that runs nanochat trainings. Exposes `train_row(row_name, jobs)` where each job is `{size: d8\|d12\|d16\|d20\|d24, parquet_key, train_seed}`. |
| `eval/` | Result aggregator (`update_all_results.py`), training-config verifier (`recompute_training_config.py`), and the canonical results JSON. |
| `emnlp_plot/` | Paper figures: cross-scale val_bpb + CORE, LLM trajectories, recipe ideaboxes. |
| `.runs/<UUID>/` | Per-WECO-run trace: `steps/<N>/files/...` is the selector source at step N; `outputs/step_<N>.out.txt` is the eval output. |
| `summary.md` | Project handoff for the next person picking this up. |

## Prerequisites

- Local 8× H100 cluster (for the WECO search loop — uses 2 GPUs per concurrent run)
- Modal account with H100:8 quota (for trainings — uses up to 5 concurrent containers)
- ClimbMix corpus downloaded to `/data/cache/nanochat/base_data_climbmix_full/`
  (6,543 parquet shards). See nanochat upstream docs for download.
- Pre-computed features on `/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_climbmix_full/`:
  `doc_tokens.npy`, `doc_chars.npy`, `distinct_{1..5}gram_bpe.npy`,
  `avg_distinct_ngram_bpe.npy`, `logppl_qwen.npy`, `topic_id.npy`,
  `format_id.npy`. See the annotation README in `data_engineering/`.

## Quick reference: which LLM produced which best recipe?

| LLM | val_bpb best step | CORE best step | Recipe family (best CORE) |
|---|---|---|---|
| GPT-5.5 | step 149 | **step 105** | 1/2 uniform + Goldilocks tri-anchor + source-block prior |
| Claude-Opus-4.7 | step 93 | step 63 | (topic, format, length-bin) cell-fair stratified quota + spam penalty |
| Gemini-3-Pro | step 112 | step 93 | 85 % uniform + length-decile Spearman-Mahalanobis quality repair |

Each selector source lives at
`.runs/<UUID>/steps/<N>/files/experiments/climbmix_full_v1/data_select_<llm>_core.py`
(or `data_select_<llm>.py` for val_bpb runs). The UUID-to-LLM mapping is in
`summary.md`.

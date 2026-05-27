# Guide 1 — Running a WECO selector search

This guide walks through running one full **WECO data-selection search**: 200
steps in which an LLM proposes a `select_docs()` function, the function is
evaluated by training nanochat-d8 on the selected documents, and WECO uses the
resulting score to inform the next proposal.

## Prerequisites

- 2 H100 GPUs free on your local node (one search uses 2 GPUs — one per
  training seed, running in parallel)
- `weco` CLI installed (`pipx install weco` or `pip install weco`)
- WECO API key in `~/.weco_credentials` (run `weco login` once)
- Anthropic / OpenAI / Google API key for whichever base LLM you pick, set as
  env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`)
- Annotation features at
  `/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_climbmix_full/`
  (see Annotation README in `data_engineering/`)
- ClimbMix shards at `/data/cache/nanochat/base_data_climbmix_full/shard_*.parquet`

## What WECO is doing

Each WECO step is a closed loop:

```
   LLM prompt          →   LLM rewrites              →   eval harness
 (instructions.md +        the data_select.py            trains nanochat-d8
  current source +         source file in place           on the selection,
  WECO history)                                           prints "core: X"
        ▲                                                         │
        └──────────────  metric fed back  ◀──────────────────────┘
```

After 200 steps the run is "complete" and WECO has a Pareto trace of every
proposal it tried.

## File layout

The CORE-optimising search (the one used in the paper's headline result) uses
these four files, all under `experiments/climbmix_full_v1/`:

| File | What it does |
|---|---|
| `data_select_<llm>_core.py` | The **source file WECO rewrites in place every step**. Holds a `select_docs(budget, seed) -> np.ndarray` function. |
| `weco_eval_full_local_core.py` | The eval harness called once per step. Imports the current `data_select_*` module, runs `select_docs`, materialises the 880 K sub-sample to `/dev/shm`, spawns 2 parallel d8 trainings, parses `CORE metric: X` from each log, and prints `core: <mean>` for WECO. |
| `weco_eval_full_local_core.sh` | Thin shell wrapper that activates the venv and invokes the harness. |
| `instructions/full_core.md` | The system prompt fed to the LLM at every step. Defines the corpus, the feature bank, the optimisation target, and anti-overfit rules. |
| `launch/base_<llm>_core_local.sh` | The launcher: sets env vars (which GPUs, which run name) then calls `weco run`. |

## Step 1 — Pick the base LLM and edit the launcher

The three CORE-max launchers shipped in this fork are:

```bash
experiments/climbmix_full_v1/launch/base_gpt55_core_local.sh        # GPUs 1,2
experiments/climbmix_full_v1/launch/base_opus47_core_local.sh       # GPUs 3,4
experiments/climbmix_full_v1/launch/base_gemini3pro_core_local.sh   # GPUs 5,6
```

A typical launcher (GPT-5.5):

```bash
export WECO_RUN_PREFIX="cmfull_base_gpt55_core_local"
export WECO_GPUS="1,2"
export WECO_DATA_SELECT_MODULE="experiments.climbmix_full_v1.data_select_gpt55_core"

weco run \
    --sources experiments/climbmix_full_v1/data_select_gpt55_core.py \
    --eval-command 'bash experiments/climbmix_full_v1/weco_eval_full_local_core.sh' \
    --metric core --goal maximize --steps 200 \
    --model gpt-5.5 \
    --eval-timeout 2400 --save-logs --output plain \
    --additional-instructions experiments/climbmix_full_v1/instructions/full_core.md
```

Edit the `WECO_GPUS` env var if you need different GPUs (the harness pins each
training seed to one GPU via `CUDA_VISIBLE_DEVICES`).

For a **val_bpb-optimising** search instead, use the non-`_core` variants:
`launch/base_<llm>.sh`, `data_select_<llm>.py`,
`weco_eval_full_local.py`, `instructions/full.md` — they replace
`--metric core --goal maximize` with `--metric val_bpb --goal minimize`.

## Step 2 — Launch

Detached (recommended — the run takes ~12 h):

```bash
cd /home/nvidia/yan/nanochat
mkdir -p /tmp/weco_core_logs
nohup bash experiments/climbmix_full_v1/launch/base_gpt55_core_local.sh \
    > /tmp/weco_core_logs/gpt55_core.log 2>&1 &
disown
```

Interactive (you'll see the LLM's plan + metric per step):

```bash
bash experiments/climbmix_full_v1/launch/base_gpt55_core_local.sh
```

### Running 3 LLMs in parallel

If you have the full 8-GPU node, launch all three CORE-max searches at once:

```bash
for llm in gpt55 opus47 gemini3pro; do
  nohup bash experiments/climbmix_full_v1/launch/base_${llm}_core_local.sh \
      > /tmp/weco_core_logs/${llm}_core.log 2>&1 &
  disown
done
```

Local GPU usage will be: GPT-5.5 on 1+2, Opus on 3+4, Gemini on 5+6. GPU 0
and 7 stay idle (GPU 0 has a thermal issue on our cluster — exclude from
sustained DDP work).

## Step 3 — Monitor

WECO prints to the launcher log file. Useful watches:

```bash
# Latest LLM-proposed plan + the most recent metric
tail -50 /tmp/weco_core_logs/gpt55_core.log

# Current step number (state file is updated after each eval)
cat .weco_iteration_cmfull_base_gpt55_core_local

# Best step found so far + its CORE
grep -oE 'best so far: [\d.]+' /tmp/weco_core_logs/gpt55_core.log | tail -1
```

The full per-step output (LLM plan + eval logs + per-task CORE breakdown)
is in `.runs/<UUID>/outputs/step_<N>.out.txt`. UUID is the run ID printed
near the top of the launcher log; you can also find it from the WECO
dashboard at `https://dashboard.weco.ai/runs/<UUID>`.

## Step 4 — Find the best step

After (or during) a run, scan all step outputs for the highest CORE:

```python
import re, pathlib, numpy as np
UUID = "<your-run-uuid>"
out_dir = pathlib.Path(f".runs/{UUID}/outputs")
rows = []
for p in sorted(out_dir.glob("step_*.out.txt")):
    m = re.search(r"^core:\s*([\d.]+)", p.read_text(), re.MULTILINE)
    if m:
        step = int(re.search(r"step_(\d+)", p.name).group(1))
        rows.append((step, float(m.group(1))))
top = sorted(rows, key=lambda r: -r[1])[:5]
for s, c in top:
    print(f"step {s:>3d}: CORE {c:.4f}")
```

The selector source for step N lives at:

```
.runs/<UUID>/steps/<N>/files/experiments/climbmix_full_v1/data_select_<llm>_core.py
```

This is the file you hand to [Guide 2](02-train-from-recipe.md) to train
nanochat at d12/d24 scale.

## Resuming a stopped or crashed run

If your launcher dies (network blip, OOM, etc.) WECO can pick up where it
left off. Use the same env vars as the original launcher, then:

```bash
yes y | weco resume <UUID> --output plain
```

(The `yes y` answers the "have source files stayed unchanged" confirmation.
If you've manually edited `data_select_*_core.py` since the run died, the
resume will start from your edited version, not the version WECO last wrote.)

Note: WECO marks a run `completed` after step 200 is recorded, even if many
of those steps were failed evals (e.g. due to disk-full crashes). Once
status is `completed`, `weco resume` will refuse. To extend further, start a
fresh run that warm-starts from the prior best selector:

```bash
# Copy the prior run's best step into the source file
cp .runs/<OLD_UUID>/steps/<BEST_STEP>/files/experiments/climbmix_full_v1/data_select_gpt55_core.py \
   experiments/climbmix_full_v1/data_select_gpt55_core.py
# Then launch with a fresh state file (different WECO_RUN_PREFIX, e.g. add _v2)
```

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `No space left on device` in step output | `/dev/shm` filled up from prior materialisations | `rm -rf /dev/shm/cmfull_base_*_core_local` between runs; resume |
| `weco resume` times out with `HTTPSConnectionPool` | WECO API transient | Just retry; we wrap in a 10-attempt loop in `launch/resume_all.sh` |
| `module ... has no attribute 'BUDGET'` | LLM rewrote `data_select_*.py` without the module-level `BUDGET` constant | Add `BUDGET = 14_374_266` back to the file and re-run; happens ~1-2% of steps |
| Eval crashes with NCCL error after warm container | Modal volume read-cache stale; not a real failure | Check `modal container exec` for real progress before killing |

## What's in `instructions/full_core.md`

The system prompt is the heart of the search. It contains:

1. The corpus description (553 M docs, 6543 shards, feature schema)
2. The selection budget (14,374,266 docs ≈ 2.6 % of corpus, ≈ 9.1 B tokens)
3. The optimisation target: **CORE (higher is better)**, with baseline values
4. The feature bank: `doc_tokens`, `doc_chars`, `distinct_{1..5}gram_bpe`,
   `avg_distinct_ngram_bpe`, `logppl_qwen`, `topic_id`, `format_id`
5. Anti-overfit rules: no hardcoded thresholds / sample sizes / category lists,
   ≤ 5 tightly-tuned numeric constants, all per-pool quantities derived at
   call time
6. The function signature contract:
   `select_docs(budget: int, seed: int) -> np.ndarray` (sorted unique int64
   indices into [0, N))

Edit `instructions/full_core.md` to:
- Try a different optimisation target (point it at a different harness)
- Restrict the feature bank (see `instructions/lexical.md`, `ppl.md`,
  `categorical.md` for ablation examples)
- Tighten or loosen the anti-overfit rules

After editing the instructions, start a fresh run — WECO caches a hash of
the instructions and won't apply them mid-run.

## Where the results land

| File | Description |
|---|---|
| `.runs/<UUID>/exec_output.jsonl` | One JSON line per step: timestamp, exit code, output_file pointer |
| `.runs/<UUID>/outputs/step_<N>.out.txt` | Full stdout of step N (LLM plan + eval log + `core: X` + per-task) |
| `.runs/<UUID>/steps/<N>/files/...` | Snapshot of the entire source tree at step N (selector file you can copy out) |
| `weco/log/<WECO_RUN_PREFIX>/cmfull_..._step<N>_s<seed>.log` | Per-seed training log for the d8 proxy run at step N |
| `.weco_iteration_<WECO_RUN_PREFIX>` | State file: integer = next step number WECO will run |

Once you have a winning step, head to [Guide 2](02-train-from-recipe.md) to
train d12 / d24 on it.

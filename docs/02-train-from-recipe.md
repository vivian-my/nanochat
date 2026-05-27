# Guide 2 — Training nanochat from an AutoData recipe

This guide turns a single `select_docs(budget, seed)` Python function
(produced by a WECO search per [Guide 1](01-weco-search.md)) into trained
nanochat models at one or more scales (d8 / d12 / d16 / d20 / d24) on Modal
H100:8.

## Pipeline overview

```
.runs/<UUID>/steps/<N>/.../data_select_*.py
                │
                ▼  (1) load + call: idx = select_docs(14_374_266, seed=42)
        14.37 M-doc index (np.int64[14_374_266])
                │
                ▼  (2) optional sub-sample for d8/d12 proxy training
        880 K / 2.31 M-doc indices (Option-B protocol)
                │
                ▼  (3) materialize: read raw ClimbMix shards,
                                    write 50 parquet buckets per index
        /dev/shm/main_baselines_parquets/<key>/bucket_*.parquet
                │
                ▼  (4) upload to Modal scratch volume
        nanochat-climbmix-scratch:main_baselines/<key>/parquets/
                │
                ▼  (5) spawn Modal train_row(s) — H100:8 DDP
        train_logs/<row_name>_<size>_<key>_s<seed>.log → val_bpb + CORE
                │
                ▼  (6) collect into eval/all_results.json
```

## Prerequisites

- Modal account + CLI (`pip install modal`, `modal token new`)
- Modal H100:8 quota (40-GPU cap = max 5 concurrent containers)
- Modal volumes already provisioned:
  - `nanochat-climbmix-scratch` (selections + logs)
  - `nanochat-archive` (raw ClimbMix + held-out val shard `shard_06542.parquet`)
- Modal app deployed: `nanochat-main-baselines` (exposes `train_row`)

  Deploy after any edit to `modal_main_baselines.py`:
  ```bash
  /home/nvidia/yan/modal_env/bin/modal deploy \
      data_engineering/modal_baselines/modal_main_baselines.py
  ```

- Local `/dev/shm` with ≥ 25 GB free per parquet set being staged

## Recipe → trained model: the easy path

The repo ships a generic builder script you can copy and edit:
`experiments/main_baselines/build_and_launch_core_gpt55_step105.py`.

It does steps 1-5 from the diagram above. Adapt it by changing the
`UUID` and `STEP` constants at the top:

```python
UUID = "56505b8e-38e5-47cb-8c16-a8c4a0eb185c"
STEP = 105
SEL_PATH = REPO / f".runs/{UUID}/steps/{STEP}/files/experiments/climbmix_full_v1/data_select_gpt55_core.py"
```

(You can also retarget a different selector file — change the filename in
`SEL_PATH`. The class of `data_select_*.py` doesn't matter as long as the
file defines `select_docs(budget, seed) -> np.ndarray` and `BUDGET = ...`.)

Then run:

```bash
cd /home/nvidia/yan/nanochat
nohup /home/nvidia/yan/modal_env/bin/python -u \
    -m experiments.main_baselines.build_and_launch_core_gpt55_step105 \
    > /tmp/recipe_build.log 2>&1 &
disown
```

Wall: ~25 min build + ~5 min upload + container time (~10 min for d8, ~10
min for d12, ~3.5 h for d24).

## Step-by-step under the hood

If you need to customise — e.g. only d24, or sweep new train_seeds — copy
the builder and edit. Here's what each section does.

### (1) Load the selector and run it

```python
import importlib.util, sys
spec = importlib.util.spec_from_file_location("sel", SEL_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules.pop(spec.name, None)
spec.loader.exec_module(mod)

full_idx = mod.select_docs(budget=14_374_266, seed=42)   # int64[14_374_266]
np.save(INDEX_DIR / f"{full_key}__index.npy", full_idx)  # cache so reruns skip the recompute
```

The selector reads features from
`/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_climbmix_full/`.
With seed=42 it's deterministic.

### (2) Sub-sample for d8 / d12 (Option-B)

For proxy training (d8 / d12) we use a fixed deterministic sub-sample of the
14.37 M selection — this keeps the data scale aligned with the proxy model's
training budget and lets us measure **selection variance** by varying the
sub-sample seed.

| size | sub-sample size | sub-sample seeds |
|---|---|---|
| d8 | 880,000 docs (~0.56 B tokens) | 42, 43, 44 (and 45, 46 for tighter stats) |
| d12 | 2,310,000 docs (~1.47 B tokens) | same |
| d16 / d20 / d24 | full 14.37 M (no sub-sample) | 1 sub-sample, 3-5 train_seeds instead |

```python
rng = np.random.default_rng(2025 + sub_seed)
sub = np.sort(rng.choice(full_idx, size=BUDGET_D12, replace=False)).astype(np.int64)
```

### (3) Materialise: read raw shards, write 50 parquet buckets

The 14.37 M index covers ~30 % of all 6543 shards. We read each contributing
shard, take only the selected rows, and group into 50 output buckets so the
downstream training reader can process them sequentially:

```python
# Find which source shards each index covers
sf = np.clip(np.searchsorted(offsets[1:], idx, side="right"), 0, n_shards - 1)
src_buckets[key] = {shard_i: (idx - offsets[shard_i]) for shard_i in np.unique(sf)}

# Per output bucket, read its source shards in parallel and concat
for out_b in range(50):
    parts = [pq.read_table(shard).take(rows) for shard, rows in ...]
    pq.write_table(pa.concat_tables(parts), staging_dir / f"bucket_{out_b:03d}.parquet")
```

This phase is I/O-bound on `/data` reads. Wall: **~15-25 min** per parquet set
on 16-thread parallel reads. Output: ~4 GB per d12 sub-sample, ~24 GB for the
full 14.37 M.

### (4) Upload to Modal

Parallel batch upload, one per key:

```python
vol = modal.Volume.from_name("nanochat-climbmix-scratch")
with vol.batch_upload(force=True) as batch:
    for f in sorted(staging_dir.iterdir()):
        batch.put_file(str(f), f"main_baselines/{key}/parquets/{f.name}")
```

Wall: ~1-2 min per d12 sub-sample (~4 GB), ~5 min for the full d24 set
(~24 GB).

### (5) Spawn Modal train_row(s)

The Modal app `nanochat-main-baselines` exposes:

```python
@app.function(gpu="H100:8", timeout=14400, ...)
def train_row(row_name: str, jobs: list[dict]) -> list[dict]:
    """Each job: {size: 'd8'|'d12'|'d16'|'d20'|'d24', parquet_key: str, train_seed: int}
    Runs jobs serially within one container; one container = one 8-GPU H100 node."""
```

Each container ships your `parquet_key` from the scratch volume into the
nanochat DDP training loop. Returns a list of `{size, parquet_key, train_seed,
val_bpb, core, exit, wall_min, log_path}`.

To spawn from your script:

```python
import modal
train_row = modal.Function.from_name("nanochat-main-baselines", "train_row")

# Example: 1 d12 + 3 parallel d24 containers
d12_jobs = [{"size": "d12", "parquet_key": key_d12(ss), "train_seed": 42}
            for ss in (42, 43, 44)]
fc_d12 = train_row.spawn("MyRecipe_d12", d12_jobs)

for ts in (42, 43, 44):
    fc = train_row.spawn(f"MyRecipe_d24_s{ts}",
                         [{"size": "d24", "parquet_key": key_full, "train_seed": ts}])
```

Save the `fc.object_id` handles to a JSON so you can poll progress without
re-spawning.

### Per-size recipes (in `modal_main_baselines.py` SIZE_RECIPE)

```python
SIZE_RECIPE = {
    "d8":  dict(depth=8,  ratio=10, device_batch_size=16, max_seq_len=1024,
                total_batch_size=131072, timeout_s=5400),
    "d12": dict(depth=12, ratio=10, device_batch_size=32, max_seq_len=2048,
                total_batch_size=524288, timeout_s=7200),
    "d16": dict(depth=16, ratio=10, device_batch_size=32, timeout_s=10800),
    "d20": dict(depth=20, ratio=10, device_batch_size=32, timeout_s=10800),
    "d24": dict(depth=24, ratio=8,  device_batch_size=16, timeout_s=14400),
}
```

`ratio` is nanochat's `--target-param-data-ratio` flag (tokens per *scaling
parameter*, see `eval/recompute_training_config.py` for the param breakdown).
d24 r=8 = ~5.84 B training tokens; this matches Karpathy's `speedrun.sh`
canonical Chinchilla-8 setting.

## Polling progress

```python
import modal, json
hs = json.loads(open("experiments/main_baselines/<your_handles>.json").read())
for h in hs:
    fc = modal.functions.FunctionCall.from_id(h["call_id"])
    try:
        r = fc.get(timeout=2)
        print(f'{h["row_name"]}: DONE  bpb={r[0]["val_bpb"]:.4f}  CORE={r[0]["core"]:.4f}')
    except TimeoutError:
        print(f'{h["row_name"]}: RUNNING')
```

For real-time training progress (Modal volume reads are cached for ~minutes
and will look "stuck"):

```bash
for cid in $(modal container list 2>&1 | grep nanochat-main | awk '{print $2}'); do
  modal container exec $cid -- /bin/bash -c "
    f=\$(ls -t /scratch/train_logs/*.log | head -1)
    grep -oE 'step [0-9]+/[0-9]+' \$f | tail -1
  "
done
```

## Collecting results

After containers finish, refresh the canonical results JSON:

```bash
/home/nvidia/yan/modal_env/bin/python -m eval.update_all_results
```

This scans every log on Modal scratch, parses final `val_bpb` + `CORE` +
per-task centered accuracies, deduplicates, computes mean ± std across seeds,
writes `eval/all_results.json`, and mirrors to
`nanochat-climbmix-scratch:eval/all_results.json`.

If your training added a new naming convention, add a regex route at the top
of `update_all_results.py` mapping the log name to `(method_name, size,
meta_dict)` so the script can route it.

## Concurrency rules (don't blow past 40 GPUs)

Modal's H100 quota in this account is **40 GPUs = 5 concurrent H100:8 containers**.
Over-provisioning causes NCCL ALLREDUCE timeouts that look like training bugs.

| Job pattern | Concurrent containers |
|---|---|
| 1 method × 3 sub_seeds d12 serial | 1 |
| 4 methods × {d12, d24} × 2 seeds, 1 method per container | 4 (each holds 4 jobs serial) |
| 6 methods × {d12, d20} × 3 seeds | 6 (over cap; works if no other Modal load) |
| All 3 LLMs × {d8, d12, d24} full sweep | Split into 2 batches of 5 |

Check current load before spawning:

```bash
modal container list 2>&1 | grep -c nanochat-main
```

## Common workflows

### Just want d24 for one recipe?

Copy `build_and_launch_gemini3p_step48_d24.py` — it's the minimal example
(no sub-sampling, just materialise full + 3 parallel d24 containers).

### Want to add more seeds to existing results?

Copy `build_and_launch_seeds45_46.py` — it reads existing cached indices
(no recompute), generates 2 new sub_seeds, and spawns 4 containers each
running 4 jobs serial.

### Want to test multiple top-K WECO steps?

Copy `build_and_launch_core_top6_d12.py` — materializes 6 selectors × 3
sub_seeds in parallel and dispatches 6 H100:8 containers.

## File summary

| File | Purpose |
|---|---|
| `data_engineering/modal_baselines/modal_main_baselines.py` | Modal app definition (`train_row`, `SIZE_RECIPE`) |
| `experiments/main_baselines/build_and_launch_*.py` | One per recipe / sweep — your "main" file |
| `experiments/main_baselines/*_handles.json` | Saved Modal call IDs for polling without re-spawning |
| `eval/update_all_results.py` | Rebuilds `all_results.json` from Modal logs |
| `eval/all_results.json` | Canonical results: per-run + summary, also mirrored on Modal |
| `summary.md` | Project-wide handoff doc with current best results |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No space left on device` mid-materialize | `/dev/shm` full | `rm -rf /dev/shm/main_baselines_parquets/<old_keys>` |
| Container "stuck at 0 bytes" via `vol.read_file` | Volume read cache, not a real hang | Use `modal container exec` to verify; almost always still training |
| `Default process group not initialized` NCCL crash | Modal H100:8 concurrency cap exceeded | Wait for prior containers, batch into ≤ 5 concurrent |
| All 3 d24 jobs in a container fail at ~25 min | Triton autotune CUDA error from FP8 + over-sized total_batch | Use nanochat defaults for d24 (no `--max-seq-len`, no `--total-batch-size` override) |
| `train_row` returns `{val_bpb: None, exit: 1}` | Look at the log: usually a `BUDGET` missing from rewritten selector | Re-add `BUDGET = 14_374_266` to the source file, re-spawn |

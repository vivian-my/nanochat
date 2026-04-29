"""One-time upload of the 100-shard ClimbMix pool + cheap-feature meta arrays
to a new Modal volume. Used by WECO v16 (cheap-features-only on 100 shards).

Volume layout:
    /shards/shard_{00000..00099}.parquet   — training pool (100 shards, ~8.6 GB)
    /shards/shard_06542.parquet            — fixed val shard (~92 MB)
    /meta/<all .npy + .json>               — cheap features only (~323 MB)

Local sources:
    /data/cache/nanochat/base_data_climbmix_full/shard_{0..99,6542}.parquet
    /data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_100shards/

Run:
    /home/nvidia/yan/modal_env/bin/modal run \\
        /home/nvidia/yan/nanochat/data_engineering/modal_upload_climbmix_100pool.py
"""
from pathlib import Path

import modal

VOLUME_NAME = "nanochat-climbmix-100pool"
SHARD_DIR   = Path("/data/cache/nanochat/base_data_climbmix_full")
META_DIR    = Path("/data/cache/nanochat/annotation/annotation_climbmix_full_v2/meta_100shards")
VAL_SHARD_IDX = 6542
N_SHARDS = 100

app = modal.App("nanochat-upload-climbmix-100pool")


@app.local_entrypoint()
def main():
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    print(f"Volume: {VOLUME_NAME}")

    upload_list = []

    for i in list(range(N_SHARDS)) + [VAL_SHARD_IDX]:
        local = SHARD_DIR / f"shard_{i:05d}.parquet"
        remote = f"/shards/shard_{i:05d}.parquet"
        if not local.exists():
            raise FileNotFoundError(local)
        upload_list.append((local, remote))

    for f in sorted(META_DIR.iterdir()):
        if f.is_file():
            upload_list.append((f, f"/meta/{f.name}"))

    with vol.batch_upload(force=True) as batch:
        for local, remote in upload_list:
            print(f"  upload {local.name} → {remote}")
            batch.put_file(str(local), remote)

    print(f"\nuploaded {len(upload_list)} files to volume `{VOLUME_NAME}`")

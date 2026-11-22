"""Download CelebA from the Hugging Face mirror and write it in the Kaggle layout.

The mirror ``flwrlabs/celeba`` stores aligned images and the 40 attributes in parquet
shards per official split. This script writes::

    data/list_attr_celeba.csv        image_id + 40 attributes as -1/1
    data/list_eval_partition.csv     image_id + partition (0 train, 1 val, 2 test)
    data/img_align_celeba/<image_id>

Image ids are assigned in shard order (``000001.png`` ...). They are unique and
consistent between the CSVs and the image files, but are not guaranteed to match
the original Kaggle file names. Landmarks and bounding boxes are not in the mirror,
so the full EDA needs the Kaggle download.

Usage::

    python scripts/download_celeba.py --data-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm import tqdm

REPO_ID = "flwrlabs/celeba"
SUBSET = "img_align+identity+attr"
SPLITS = {"train": 0, "valid": 1, "test": 2}
NON_ATTRIBUTE_COLUMNS = {"image", "celeb_id"}


def download(data_dir: Path) -> None:
    attr_path = data_dir / "list_attr_celeba.csv"
    part_path = data_dir / "list_eval_partition.csv"
    if attr_path.is_file() and part_path.is_file():
        print(f"[download] {data_dir} already prepared; skipping")
        return

    cache = Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=[f"{SUBSET}/*.parquet"]))
    image_dir = data_dir / "img_align_celeba"
    image_dir.mkdir(parents=True, exist_ok=True)

    attr_rows, part_rows = [], []
    next_id = 1
    for split, code in SPLITS.items():
        shards = sorted((cache / SUBSET).glob(f"{split}-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards for split '{split}' in {cache / SUBSET}")
        for shard in tqdm(shards, desc=f"Writing {split}"):
            table = pq.read_table(shard).to_pandas()
            attributes = [c for c in table.columns if c not in NON_ATTRIBUTE_COLUMNS]
            for row in table.itertuples(index=False):
                record = row._asdict()
                image = record["image"]
                suffix = Path(image.get("path") or "x.png").suffix or ".png"
                image_id = f"{next_id:06d}{suffix}"
                (image_dir / image_id).write_bytes(image["bytes"])
                attr_rows.append({"image_id": image_id, **{a: 1 if record[a] else -1 for a in attributes}})
                part_rows.append({"image_id": image_id, "partition": code})
                next_id += 1

    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(attr_rows).to_csv(attr_path, index=False)
    pd.DataFrame(part_rows).to_csv(part_path, index=False)
    print(f"[download] wrote {next_id - 1} images to {image_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    download(parser.parse_args(argv).data_dir)


if __name__ == "__main__":
    main()

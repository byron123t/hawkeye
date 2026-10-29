"""
Reading the BSTRB dataset in its Hugging Face layout.

    <root>/data/{train,val,test}/metadata.jsonl      one JSON object per image
    <root>/data/{split}/<file_name>                  the image

Fields of a row: id, split, kind ("video_frame" | "photo"), video_id (frames only), frame
(position in the video's capture, frames only), text_height_px (mean text height, frames
only), file_name, prompt, transcription, label_source ("human" | "auto").

`root` is a local copy (e.g. from `huggingface-cli download byron123t/BSTRB --repo-type dataset
--local-dir BSTRB`); `resolve_root()` downloads it if given the repo id instead. The dataset is
gated: accept the terms on its Hugging Face page and log in first.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, List, Optional

DATASET_ID = "byron123t/BSTRB"
SPLITS = ("train", "val", "test")


def resolve_root(root: Optional[str] = None) -> Path:
    """A local dataset directory; downloads the dataset when `root` is a repo id or unset."""
    root = root or os.environ.get("BSTRB_ROOT") or DATASET_ID
    p = Path(root)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=root, repo_type="dataset"))


def iter_split(root: str | Path, split: str, kinds: Optional[set] = None) -> Iterator[dict]:
    """Rows of one split with `image` set to the image's absolute path."""
    d = Path(root) / "data" / split
    with open(d / "metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if kinds and r.get("kind") not in kinds:
                continue
            r["image"] = str((d / r["file_name"]).resolve())
            yield r


def load_split(root: str | Path, split: str, kinds: Optional[set] = None) -> List[dict]:
    return list(iter_split(root, split, kinds))


def unit_of(row: dict) -> str:
    """Resampling unit for the statistics: the video for a frame, the image for a photo."""
    return row.get("video_id") or row["id"]


__all__ = ["DATASET_ID", "SPLITS", "resolve_root", "iter_split", "load_split", "unit_of"]

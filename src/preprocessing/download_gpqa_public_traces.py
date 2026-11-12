"""Download separate GPQA solution sources, without retaining sxiong hidden reasoning.

Run: .venv/bin/python src/preprocessing/download_gpqa_public_traces.py
All downloaded data and caches stay under data/source/gpqa_reasoning_sources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi, get_token


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/source/gpqa_reasoning_sources"
SXIONG_COLUMNS = (
    "id", "problem_id", "domain", "subdomain", "question", "explanation", "answer",
    "content", "correct", "model", "source_split",
)


def url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(session: requests.Session, address: str, path: Path) -> Path:
    with session.get(address, stream=True, timeout=120) as response:
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                out.write(chunk)
        tmp.replace(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    token = get_token()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    api = HfApi()
    manifest = {"sources": {}, "sxiong_hidden_reasoning_persisted": False}

    repo = "Idavidrein/gpqa"
    revision = api.dataset_info(repo).sha
    official = OUT / "original/gpqa_extended.csv"
    try:
        download(session, url(repo, revision, "gpqa_extended.csv"), official)
    except requests.HTTPError as exc:
        if exc.response.status_code not in (401, 403):
            raise
        print("Official GPQA data requires authentication; using the public explanation-preserving release.", flush=True)
        mirror = "LLMcompe-Team-Watanabe/gpqa_preprocess"
        mirror_revision = api.dataset_info(mirror).sha
        name = "data/test_gpqa_extended-00000-of-00001.parquet"
        path = download(session, url(mirror, mirror_revision, name), OUT / "original/gpqa_extended_public.parquet")
        df = pd.read_parquet(path)
        manifest["sources"]["original"] = {"repo": repo, "revision": revision,
            "download_repo": mirror, "download_revision": mirror_revision,
            "filename": name, "local_file": str(path.relative_to(ROOT)), "rows": len(df),
            "sha256": sha256(path), "official_requires_authentication": True}
    else:
        df = pd.read_csv(official)
        manifest["sources"]["original"] = {"repo": repo, "revision": revision,
            "filename": "gpqa_extended.csv", "local_file": str(official.relative_to(ROOT)),
            "rows": len(df), "sha256": sha256(official), "official_requires_authentication": False}
    print(f"Original explanation source: {len(df)} rows, columns={list(df.columns)}", flush=True)

    repo = "sxiong/GPQA-trajectory"
    revision = api.dataset_info(repo).sha
    rows = []
    counts = {}
    for split in ("train", "test"):
        count = 0
        # Parse the remote stream in memory. The downloaded JSONL is never written,
        # so reasoning_content is not retained even as a temporary/cache file.
        with session.get(url(repo, revision, f"{split}.jsonl"), stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                raw = json.loads(line)
                selected = {k: raw.get(k) for k in SXIONG_COLUMNS if k != "source_split"}
                selected["source_split"] = split
                rows.append(selected)
                count += 1
        counts[split] = count
        print(f"sxiong {split}: retained {count} written solutions", flush=True)
    written = pd.DataFrame(rows, columns=SXIONG_COLUMNS)
    path = OUT / "sxiong_written_solutions.parquet"
    written.to_parquet(path, index=False)
    assert "reasoning_content" not in written.columns
    manifest["sources"]["sxiong"] = {"repo": repo, "revision": revision,
        "local_file": str(path.relative_to(ROOT)), "rows": len(written),
        "split_rows": counts, "retained_columns": list(written.columns), "sha256": sha256(path)}

    repo = "spawn99/GPQA-diamond-ClaudeR1"
    revision = api.dataset_info(repo).sha
    name = "data/train-00000-of-00001.parquet"
    path = download(session, url(repo, revision, name), OUT / "spawn99_solutions.parquet")
    spawn = pd.read_parquet(path)
    manifest["sources"]["spawn99"] = {"repo": repo, "revision": revision,
        "filename": name, "local_file": str(path.relative_to(ROOT)), "rows": len(spawn), "sha256": sha256(path)}
    print(f"spawn99: {len(spawn)} rows, columns={list(spawn.columns)}", flush=True)
    (OUT / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved manifest to {OUT / 'download_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

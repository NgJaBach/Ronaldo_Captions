"""The academy. Turns every caption into a vector so we can search by meaning, not just by words.

Usage (repo root):
  python src/embed.py            # embeds only new/changed captions, so it is safe to re-run
  python src/embed.py --rebuild  # start from scratch

Writes data/embeddings.npy (float16, L2-normalised) and data/embeddings.json (model + code/hash per row).
Model: intfloat/multilingual-e5-small (118M params, CPU is fine, handles English/Portuguese/Arabic/...).
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "ronaldo.db"
NPY = ROOT / "data" / "embeddings.npy"
META = ROOT / "data" / "embeddings.json"
MODEL = "intfloat/multilingual-e5-small"


def caption_hash(c: str) -> str:
    return hashlib.sha1(c.encode("utf-8")).hexdigest()[:12]


def main(rebuild: bool = False) -> None:
    from sentence_transformers import SentenceTransformer

    con = sqlite3.connect(DB)
    posts = con.execute("SELECT code, caption FROM posts WHERE caption != '' ORDER BY code").fetchall()

    old_vec, old_rows = None, {}
    if not rebuild and NPY.exists() and META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        if meta["model"] == MODEL:
            old_vec = np.load(NPY)
            old_rows = {r["code"]: (i, r["hash"]) for i, r in enumerate(meta["rows"])}

    todo = [(c, cap) for c, cap in posts if c not in old_rows or old_rows[c][1] != caption_hash(cap)]
    print(f"{len(posts)} posts with captions, {len(todo)} to embed")

    new_vecs = {}
    if todo:
        model = SentenceTransformer(MODEL, device="cpu")
        vecs = model.encode(["passage: " + cap for _, cap in todo], batch_size=32,
                            normalize_embeddings=True, show_progress_bar=True)
        new_vecs = {c: v for (c, _), v in zip(todo, vecs)}

    rows, mat = [], []
    for code, cap in posts:
        if code in new_vecs:
            mat.append(new_vecs[code])
        else:
            mat.append(old_vec[old_rows[code][0]].astype(np.float32))
        rows.append({"code": code, "hash": caption_hash(cap)})

    np.save(NPY, np.asarray(mat, dtype=np.float16))
    META.write_text(json.dumps({"model": MODEL, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(rows)} vectors to {NPY.name}. The academy has graduated another class.")


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv)

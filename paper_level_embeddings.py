"""Builds two auxiliary paper-level embeddings, as an ALTERNATIVE paper-ranking
signal to the pipeline's default (max raw dense score over a paper's artifacts):

  1. abstract embedding: title + abstract text, embedded directly (same way any
     other document-side artifact is embedded -- no query instruction prefix).
  2. mean-pool embedding: the centroid of every one of that paper's artifact
     embeddings (passages/tables/figures), re-normalized to unit length.

Neither replaces the production ranking in answer.py -- these are ablations,
compared against the baseline in paper_ranking_ablation.py.

"""
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import common

OUT_PATH = common.INDEX_DIR / "paper_level_embeddings.npz"


def main():
    papers = common.load_papers()
    catalog = json.loads((common.INDEX_DIR / "catalog.json").read_text())
    index = faiss.read_index(str(common.INDEX_DIR / "artifacts.faiss"))

    # Reconstruct every artifact's stored (normalized) embedding, in the same
    # order as catalog.json, so we can mean-pool them per paper_id.
    all_embs = index.reconstruct_n(0, index.ntotal)

    paper_ids = [p["paper_id"] for p in papers]

    # --- mean-pool ---
    meanpool = np.zeros((len(paper_ids), all_embs.shape[1]), dtype="float32")
    for i, pid in enumerate(paper_ids):
        rows = [all_embs[j] for j, a in enumerate(catalog) if a["paper_id"] == pid]
        if not rows:
            continue
        v = np.mean(rows, axis=0)
        norm = np.linalg.norm(v)
        meanpool[i] = v / norm if norm > 0 else v

    # --- title + abstract ---
    texts = []
    for pid in paper_ids:
        full_text = common.paper_paths(pid)["text"].read_text(encoding="utf-8", errors="ignore")
        title, abstract = common.extract_title_abstract(full_text)
        texts.append(f"{title}\n{abstract}" if abstract else title)

    print("Embedding title+abstract for each paper with BAAI/bge-large-en-v1.5 ...")
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
    abstract_embs = embedder.encode(texts, normalize_embeddings=True).astype("float32")

    np.savez(
        OUT_PATH,
        paper_ids=np.array(paper_ids),
        abstract_embs=abstract_embs,
        meanpool_embs=meanpool,
    )
    print(f"Saved paper-level embeddings for {len(paper_ids)} papers to {OUT_PATH}")


if __name__ == "__main__":
    main()

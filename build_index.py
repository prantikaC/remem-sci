"""Builds the evidence-fragment index: passages, tables, and figures across the
15 curated papers, embedded with bge-large-en-v1.5 into one FAISS flat index.
Figure descriptions are generated once with Qwen2.5-VL-7B-Instruct and cached.
"""
import json
import re

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

import common

FIG_DESC_PROMPT = (
    "Describe this figure from a machine learning research paper in 2-4 sentences. "
    "Focus on what would help someone recognize it later purely from memory: the type of "
    "chart or diagram (bar chart, line plot, architecture diagram, scatter plot, schema, "
    "screenshot, etc.), its overall shape/trend if it's a plot, and its structural layout "
    "if it's a diagram. Do not restate the caption verbatim; describe what the image itself "
    "visually shows."
)


def build_figure_caption(vlm_generate, image_path, real_caption):
    img = Image.open(image_path).convert("RGB")
    desc = vlm_generate(FIG_DESC_PROMPT, images=[img])
    return f"{real_caption}\nVisual description: {desc}"


def main():
    common.INDEX_DIR.mkdir(exist_ok=True)
    cache_path = common.INDEX_DIR / "figure_caption_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    papers = common.load_papers()
    catalog = []

    # Lazy VLM import/load only if we actually have new figures to caption.
    vlm_generate = None

    for p in papers:
        pid = p["paper_id"]
        paths = common.paper_paths(pid)
        full_text = paths["text"].read_text(encoding="utf-8", errors="ignore")

        # --- passages ---
        for i, ch in enumerate(common.chunk_full_text(full_text)):
            aid = common.artifact_id(pid, "passage", f"{i:03d}")
            catalog.append({
                "artifact_id": aid,
                "paper_id": pid,
                "type": "passage",
                "content_for_embedding": ch["chunk_text"][:2000],
                "display_content": ch["chunk_text"],
                "local_context": ch["header"],
            })

        # --- tables ---
        if paths["tables_dir"].exists():
            for tpath in sorted(paths["tables_dir"].glob("table_*.csv")):
                num = int(re.search(r"(\d+)", tpath.stem).group(1))
                aid = common.artifact_id(pid, "table", f"{num:03d}")
                caption = common.find_caption(full_text, "Table", num)
                md = common.csv_to_markdown(tpath)
                catalog.append({
                    "artifact_id": aid,
                    "paper_id": pid,
                    "type": "table",
                    "content_for_embedding": f"{caption}\n{md}"[:2000],
                    "display_content": f"{caption}\n\n{md}",
                    "local_context": caption,
                    "source_path": str(tpath.relative_to(common.ROOT)),
                })

        # --- figures ---
        if paths["figures_dir"].exists():
            for fpath in sorted(paths["figures_dir"].glob("figure_*.png")):
                num = int(re.search(r"(\d+)", fpath.stem).group(1))
                aid = common.artifact_id(pid, "figure", f"{num:03d}")
                caption = common.find_caption(full_text, "Figure", num)
                if aid in cache:
                    content = cache[aid]
                else:
                    if vlm_generate is None:
                        import vlm_utils
                        vlm_generate = vlm_utils.generate
                        print("Loaded Qwen2.5-VL-7B-Instruct for figure captioning.")
                    print(f"Captioning {aid} ...")
                    content = build_figure_caption(vlm_generate, fpath, caption)
                    cache[aid] = content
                    cache_path.write_text(json.dumps(cache, indent=2))
                catalog.append({
                    "artifact_id": aid,
                    "paper_id": pid,
                    "type": "figure",
                    "content_for_embedding": content[:2000],
                    "display_content": content,
                    "local_context": caption,
                    "source_path": str(fpath.relative_to(common.ROOT)),
                })

        n_pass = sum(1 for a in catalog if a["paper_id"] == pid and a["type"] == "passage")
        n_tab = sum(1 for a in catalog if a["paper_id"] == pid and a["type"] == "table")
        n_fig = sum(1 for a in catalog if a["paper_id"] == pid and a["type"] == "figure")
        print(f"{pid}: {n_pass} passages, {n_tab} tables, {n_fig} figures")

    print(f"\nTotal artifacts: {len(catalog)}")

    print("Embedding artifacts with BAAI/bge-large-en-v1.5 ...")
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
    texts = [a["content_for_embedding"] for a in catalog]
    embs = embedder.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    embs = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(common.INDEX_DIR / "artifacts.faiss"))

    for a in catalog:
        a.pop("content_for_embedding", None)
    (common.INDEX_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2))

    print(f"Saved FAISS index and catalog for {len(catalog)} artifacts to {common.INDEX_DIR}")


if __name__ == "__main__":
    main()

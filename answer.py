"""Shared retrieval + grounded-generation pipeline: one code path for every
query, regardless of which memory 'case' it belongs to (the case is only used
later, in evaluate.py, as evaluation metadata).

PAPER-FIRST, HIERARCHICAL pipeline (paper identification gates evidence search):
q -> embed -> dense search over the FULL corpus (all papers)
   -> aggregate to a coarse paper ranking (each paper's best raw dense score)
   -> shortlist the top-K papers ONLY
   -> restrict candidates to those papers' artifacts, cross-encoder rerank -> top-5
   -> feed all top-5 artifacts' local context (+ images for any figures)
   -> Qwen2.5-VL-7B-Instruct grounded answer generation, citing which
      evidence item it actually used

The reported paper ranking is the coarse, PRE-rerank ranking (decided before any
evidence-level search), not derived post-hoc from which papers the top evidence
happened to belong to -- so Paper Recall and Evidence Recall are now genuinely
independent measurements of two sequential stages, not two views of one flat search.
"""
import json
import re

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import CrossEncoder, SentenceTransformer

import common
import vlm_utils

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_embedder = None
_reranker = None
_index = None
_catalog = None
_paper_meta = common.load_paper_meta()


def load_all():
    global _embedder, _reranker, _index, _catalog
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda:0")
        _reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda:0")
        _index = faiss.read_index(str(common.INDEX_DIR / "artifacts.faiss"))
        _catalog = json.loads((common.INDEX_DIR / "catalog.json").read_text())
    return _embedder, _reranker, _index, _catalog


def retrieve(query, top_k=30):
    """Flat top-k over the whole corpus. Retained for the artifact-level cross-encoder
    step and for standalone use, but no longer what paper ranking is derived from."""
    embedder, _, index, catalog = load_all()
    qvec = embedder.encode(
        [BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).astype("float32")
    scores, idxs = index.search(qvec, top_k)
    return [(float(s), catalog[i]) for s, i in zip(scores[0], idxs[0]) if i != -1]


def retrieve_full(query):
    """Dense search over EVERY artifact in the corpus (not just a truncated top-k),
    so the paper-level ranking below reflects each paper's true best match rather
    than being bottlenecked by whatever a small top-k happened to include."""
    embedder, _, index, catalog = load_all()
    qvec = embedder.encode(
        [BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).astype("float32")
    scores, idxs = index.search(qvec, index.ntotal)
    return [(float(s), catalog[i]) for s, i in zip(scores[0], idxs[0]) if i != -1]


def rank_papers(full_candidates):
    """Coarse, PRE-rerank paper ranking: each paper's rank is its best raw dense
    score across ALL of its artifacts. This is the paper-identification stage,
    decided before any evidence-level search happens."""
    best = {}
    for score, art in full_candidates:
        pid = art["paper_id"]
        if pid not in best or score > best[pid]:
            best[pid] = score
    return [pid for pid, _ in sorted(best.items(), key=lambda kv: -kv[1])]


def rerank(query, candidates, top_n=5):
    _, reranker, _, _ = load_all()
    pairs = [(query, c["display_content"][:1500]) for _, c in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(-scores)
    return [(float(scores[i]), candidates[i][1]) for i in order[:top_n]]


ANSWER_PROMPT = """You are helping a researcher who half-remembers something from a paper, sometimes from more than one paper, or possibly misremembering details from two different
papers as if they were one. You are given up to 5 candidate pieces of evidence retrieved for
their question -- they are NOT all guaranteed to be relevant, and they may come from different
papers. Identify which evidence item(s) actually answer the question, and answer using ONLY
that evidence.

If the question asks you to compare, or the evidence reveals the researcher is blending
details from two different papers, say so explicitly and address each paper separately rather
than forcing a single answer.

Answer concisely (2-6 sentences), citing concrete numbers/names from the evidence when
relevant, and mentioning where in the paper it comes from (e.g. which table, figure, or
section) when that's given in the evidence's context, so the researcher could look it up
themselves. If none of the evidence supports an answer, say so rather than guessing. Do NOT
try to name any paper's title or authors yourself -- citations will be attached automatically
from metadata afterward.

On its own final line, write exactly: SOURCE_EVIDENCE: <n>
or, if your answer draws on evidence from more than one paper, list all the evidence numbers
you relied on separated by commas, e.g. SOURCE_EVIDENCE: 1,3
Write NONE if you could not answer from any of the evidence.

Question: {query}

{evidence_block}
"""

EVIDENCE_ITEM = """Evidence {n} (paper_id: {paper_id}, evidence type: {etype}, context: "{context}"):
{content}
"""

SOURCE_LINE_RE = re.compile(r"SOURCE_EVIDENCE:\s*([\d,\s]+|NONE)", re.IGNORECASE)


def generate_answer(query, top_artifacts):
    """top_artifacts: list of catalog dicts, highest-ranked first (up to 5).
    Returns (answer_text_without_source_line, list_of_source_evidence_indices)."""
    blocks = []
    images = []
    for i, art in enumerate(top_artifacts, start=1):
        blocks.append(EVIDENCE_ITEM.format(
            n=i,
            paper_id=art["paper_id"],
            etype=art["type"],
            context=art["local_context"],
            content=art["display_content"][:1200],
        ))
        if art["type"] == "figure" and art.get("source_path"):
            images.append(Image.open(common.ROOT / art["source_path"]).convert("RGB"))
    evidence_block = "\n".join(blocks)
    prompt = ANSWER_PROMPT.format(query=query, evidence_block=evidence_block)
    raw = vlm_utils.generate(prompt, images=images or None, max_new_tokens=400)

    m = SOURCE_LINE_RE.search(raw)
    source_indices = []
    if m and m.group(1).strip().upper() != "NONE":
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(top_artifacts):
                    source_indices.append(idx)
    answer_text = SOURCE_LINE_RE.sub("", raw).strip()
    return answer_text, source_indices


def answer_query(query, paper_shortlist_k=5, within_paper_top_k=30, rerank_n=5):
    # Stage 1 -- PAPER IDENTIFICATION: rank every paper by its best raw dense
    # score, using the full corpus (not a truncated top-k), before any evidence
    # search happens.
    full_candidates = retrieve_full(query)
    ranked_papers = rank_papers(full_candidates)
    shortlist = set(ranked_papers[:paper_shortlist_k])

    # Stage 2 -- EVIDENCE LOCALIZATION: search ONLY within the shortlisted
    # papers' artifacts (evidence search is gated by the paper stage, not run
    # independently over the whole corpus).
    within_shortlist = [c for c in full_candidates if c[1]["paper_id"] in shortlist]
    within_shortlist = within_shortlist[:within_paper_top_k]
    reranked = rerank(query, within_shortlist, top_n=rerank_n)
    top_artifacts = [a for _, a in reranked]
    answer_text, source_indices = generate_answer(query, top_artifacts)

    # Attach citation(s) deterministically from metadata -- never from the LLM's
    # own (unreliable) recall of titles/authors -- for whichever paper(s) it says
    # it actually used, falling back to the top-ranked paper if it didn't say.
    if source_indices:
        source_paper_ids = list(dict.fromkeys(top_artifacts[i]["paper_id"] for i in source_indices))
    else:
        source_paper_ids = [ranked_papers[0]]
    citations = [
        common.format_citation(_paper_meta[pid]) if pid in _paper_meta else pid
        for pid in source_paper_ids
    ]
    if len(citations) == 1:
        full_answer = f"{answer_text}\n\nSource: {citations[0]}"
    else:
        sources_block = "\n".join(f"- {c}" for c in citations)
        full_answer = f"{answer_text}\n\nSources:\n{sources_block}"

    return {
        "query": query,
        "ranked_papers": ranked_papers,
        "ranked_papers_citations": [
            common.format_citation(_paper_meta[pid]) if pid in _paper_meta else pid
            for pid in ranked_papers
        ],
        "top_artifacts": [
            {"artifact_id": a["artifact_id"], "score": s, "paper_id": a["paper_id"], "type": a["type"]}
            for s, a in reranked
        ],
        "top_artifact_id": top_artifacts[0]["artifact_id"],
        "source_paper_ids": source_paper_ids,
        "generated_answer": full_answer,
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval-augmented generation?"
    result = answer_query(q)
    print(json.dumps(result, indent=2))

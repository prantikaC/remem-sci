"""Ablation: compares three paper-ranking strategies against the same QA set,
using ONLY the paper-identification stage (no reranker, no VLM generation --
this is a pure retrieval-stage comparison, so it's fast and GPU-light):

  baseline  -- production ranking in answer.rank_papers(): each paper's score
               is the single best raw dense score across all of its artifacts.
  abstract  -- each paper's score is the query's dense similarity to that
               paper's title+abstract embedding (paper_level_embeddings.py).
  meanpool  -- each paper's score is the query's dense similarity to the
               centroid (mean) of all of that paper's artifact embeddings.

"""
import json

import numpy as np

import answer
import common
from evaluate import gold_ids_of

PAPER_LEVEL_PATH = common.INDEX_DIR / "paper_level_embeddings.npz"


def load_paper_level():
    data = np.load(PAPER_LEVEL_PATH, allow_pickle=True)
    paper_ids = list(data["paper_ids"])
    return paper_ids, data["abstract_embs"], data["meanpool_embs"]


def rank_by_paper_level(qvec, paper_ids, embs):
    scores = embs @ qvec  # both sides are unit-normalized -> inner product = cosine
    order = np.argsort(-scores)
    return [paper_ids[i] for i in order]


def main(qa_name="data/qa_set.json"):
    qa_items = common.load_qa(qa_name)
    paper_ids, abstract_embs, meanpool_embs = load_paper_level()

    embedder, _, _, _ = answer.load_all()

    rows = []
    for qa in qa_items:
        gold_ids = gold_ids_of(qa)
        qvec = embedder.encode(
            [answer.BGE_QUERY_INSTRUCTION + qa["query_text"]], normalize_embeddings=True
        ).astype("float32")[0]

        full_candidates = answer.retrieve_full(qa["query_text"])
        baseline_ranking = answer.rank_papers(full_candidates)
        abstract_ranking = rank_by_paper_level(qvec, paper_ids, abstract_embs)
        meanpool_ranking = rank_by_paper_level(qvec, paper_ids, meanpool_embs)

        rows.append({
            "id": qa["id"], "case": qa["case"], "gold_ids": gold_ids,
            "baseline": baseline_ranking, "abstract": abstract_ranking,
            "meanpool": meanpool_ranking,
        })

    def hit_at(ranking, gold_ids, k):
        if k == 1:
            return bool(ranking) and ranking[0] in gold_ids
        return all(g in ranking[:k] for g in gold_ids)

    strategies = ["baseline", "abstract", "meanpool"]
    cases = sorted(set(r["case"] for r in rows))

    print(f"{'Case':<8}{'N':<4}" + "".join(f"{s+' R@1':<14}{s+' R@5':<14}" for s in strategies))
    summary = {"by_case": {}, "cases_1_4": {}, "all": {}}
    for case in cases:
        crows = [r for r in rows if r["case"] == case]
        n = len(crows)
        line = f"{str(case):<8}{n:<4}"
        summary["by_case"][str(case)] = {"n": n}
        for s in strategies:
            r1 = sum(hit_at(r[s], r["gold_ids"], 1) for r in crows) / n
            r5 = sum(hit_at(r[s], r["gold_ids"], 5) for r in crows) / n
            summary["by_case"][str(case)][s] = {"r1": r1, "r5": r5}
            line += f"{r1:<14.2f}{r5:<14.2f}"
        print(line)

    for label, subset in [("Cases1-4", [r for r in rows if r["case"] in (1, 2, 3, 4)]),
                           ("All", rows)]:
        n = len(subset)
        line = f"{label:<8}{n:<4}"
        key = "cases_1_4" if label == "Cases1-4" else "all"
        summary[key] = {"n": n}
        for s in strategies:
            r1 = sum(hit_at(r[s], r["gold_ids"], 1) for r in subset) / n
            r5 = sum(hit_at(r[s], r["gold_ids"], 5) for r in subset) / n
            summary[key][s] = {"r1": r1, "r5": r5}
            line += f"{r1:<14.2f}{r5:<14.2f}"
        print(line)

    out_path = common.RESULTS_DIR / "paper_ranking_ablation_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import sys
    qa_arg = sys.argv[1] if len(sys.argv) > 1 else "data/qa_set.json"
    main(qa_arg)

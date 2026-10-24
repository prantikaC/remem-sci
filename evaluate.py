"""Runs every QA item through the single shared pipeline (answer.answer_query),
with no case-conditioning, then reports Paper Identification / Evidence
Localization Recall@1/@5 per case, plus generated vs. reference answers for a
manual Answer Grounding pass.

"""
import json

import answer
import common


def gold_ids_of(qa):
    """[] (present but empty) means deliberately ambiguous -- no single correct
    paper, so paper/evidence recall are not applicable for that item."""
    if "gold_paper_ids" in qa:
        return qa["gold_paper_ids"]
    return [qa["gold_paper_id"]]


def evidence_hit(top_artifacts, catalog_by_id, qa, n):
    top = top_artifacts[:n]
    gold_ids = gold_ids_of(qa)
    if qa.get("gold_artifact_id"):
        return any(a["artifact_id"] == qa["gold_artifact_id"] for a in top)
    kws = qa.get("gold_keywords")
    if kws:
        kws = [k.lower() for k in kws]
        for a in top:
            if a["paper_id"] not in gold_ids or a["type"] != "passage":
                continue
            content = catalog_by_id[a["artifact_id"]]["display_content"].lower()
            if any(k in content for k in kws):
                return True
        return False
    kws_by_paper = qa.get("gold_keywords_by_paper")
    if kws_by_paper:
        satisfied = set()
        for a in top:
            pid = a["paper_id"]
            if pid not in kws_by_paper:
                continue
            content = catalog_by_id[a["artifact_id"]]["display_content"].lower()
            if any(k.lower() in content for k in kws_by_paper[pid]):
                satisfied.add(pid)
        return satisfied == set(kws_by_paper.keys())
    # Multi-paper items with no keyword spec at all: require at least one
    # artifact from EACH gold paper to have surfaced in the top-n.
    covered = {a["paper_id"] for a in top if a["paper_id"] in gold_ids}
    return covered == set(gold_ids)


def main(out_name="results/results.json", qa_name="data/qa_set.json"):
    qa_items = common.load_qa(qa_name)
    catalog = json.loads((common.INDEX_DIR / "catalog.json").read_text())
    catalog_by_id = {a["artifact_id"]: a for a in catalog}

    results = []
    for qa in qa_items:
        gold_ids = gold_ids_of(qa)
        print(f"\n=== {qa['id']} (case {qa['case']}) ===")
        print("Q:", qa["query_text"])
        res = answer.answer_query(qa["query_text"])

        if gold_ids:
            paper_r1 = bool(res["ranked_papers"]) and res["ranked_papers"][0] in gold_ids
            paper_r5 = all(g in res["ranked_papers"][:5] for g in gold_ids)
            ev_r1 = evidence_hit(res["top_artifacts"], catalog_by_id, qa, 1)
            ev_r5 = evidence_hit(res["top_artifacts"], catalog_by_id, qa, 5)
            print("Predicted top paper:", res["ranked_papers"][0] if res["ranked_papers"] else None,
                  "| gold(s):", gold_ids, "| R@1 (top-1 is a gold paper):", paper_r1,
                  "R@5 (ALL gold papers present):", paper_r5)
            print("Top artifact:", res["top_artifact_id"], "| evidence R@1:", ev_r1, "R@5:", ev_r5)
        else:
            paper_r1 = paper_r5 = ev_r1 = ev_r5 = None
            print("Deliberately ambiguous item -- no single gold paper; skipping paper/evidence recall.")
            print("Ranked papers surfaced:", res["ranked_papers"][:5])
        print("Generated answer:", res["generated_answer"])
        print("Reference answer:", qa["reference_answer"])

        results.append({
            "id": qa["id"], "case": qa["case"], "query": qa["query_text"],
            "gold_paper_ids": gold_ids, "gold_artifact_id": qa["gold_artifact_id"],
            "predicted_ranked_papers": res["ranked_papers"],
            "predicted_top_artifact": res["top_artifact_id"],
            "predicted_top5_artifacts": [a["artifact_id"] for a in res["top_artifacts"]],
            "paper_recall_at_1": paper_r1, "paper_recall_at_5": paper_r5,
            "evidence_recall_at_1": ev_r1, "evidence_recall_at_5": ev_r5,
            "cited_paper_ids": res["source_paper_ids"],
            "generated_answer": res["generated_answer"],
            "reference_answer": qa["reference_answer"],
            "qualitative": qa.get("qualitative", False),
            "manual_grounding_pass": None,
        })

    def fmt(v):
        return f"{v:.2f}" if v is not None else "n/a"

    metrics = ["paper_recall_at_1", "paper_recall_at_5", "evidence_recall_at_1", "evidence_recall_at_5"]
    summary = {"by_case": {}, "all": {}}

    print("\n\n================ SUMMARY BY CASE ================")
    print(f"{'Case':<12}{'N':<4}{'PaperR@1':<10}{'PaperR@5':<10}{'EvidR@1':<10}{'EvidR@5':<10}")
    for case in [1, 2, 3, 4, 5, 6]:
        rows = [r for r in results if r["case"] == case]
        n = len(rows)
        if n == 0:
            continue
        vals = {m: common.avg(rows, m) for m in metrics}
        summary["by_case"][str(case)] = {"n": n, **vals}
        print(f"{str(case):<12}{n:<4}{fmt(vals['paper_recall_at_1']):<10}{fmt(vals['paper_recall_at_5']):<10}"
              f"{fmt(vals['evidence_recall_at_1']):<10}{fmt(vals['evidence_recall_at_5']):<10}")

    all_vals = {m: common.avg(results, m) for m in metrics}
    summary["all"] = {"n": len(results), **all_vals}
    print(f"{'ALL':<12}{len(results):<4}{fmt(all_vals['paper_recall_at_1']):<10}{fmt(all_vals['paper_recall_at_5']):<10}"
          f"{fmt(all_vals['evidence_recall_at_1']):<10}{fmt(all_vals['evidence_recall_at_5']):<10}")
    print("(ambiguous-case items are excluded from recall averages -- N/A by design, see grounding_check.py instead)")

    out_path = common.ROOT / out_name
    out_path.write_text(json.dumps(results, indent=2))
    summary_path = common.summary_path_for(out_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved detailed results to {out_path}")
    print(f"Saved aggregate summary to {summary_path}")
    print("Next: python grounding_check.py <results> to add objective grounding, then answer_correct.py.")


if __name__ == "__main__":
    import sys
    out_arg = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
    qa_arg = sys.argv[2] if len(sys.argv) > 2 else "data/qa_set.json"
    main(out_arg, qa_arg)

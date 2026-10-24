"""Computes the final Answer Correct metric for a results file.

Cases 1-5: correct iff the generated answer matches the gold answer factually
(objective grounding verdict == "pass"; partial counts as incorrect) AND the
papers it cites are exactly the gold paper(s).

Case 6 (inaccurate memory): correct iff the answer detects the false premise --
it either states the claim is not supported by the evidence, or explicitly
corrects it with accurate information. Confirming the false premise is
incorrect. This needs reading the answer, so it comes from
data/manual_judgments.json, which also holds any documented overrides of the
automatic rule for Cases 1-5 (e.g. a grounding "pass" earned only because the
appended citation line contains a required name).

Usage: python answer_correct.py results/results_v20_final.json
"""
import json
import re
import sys

import common

MANUAL_PATH = common.DATA_DIR / "manual_judgments.json"


def cited_paper_ids(item, title_to_pid):
    if "cited_paper_ids" in item:
        return set(item["cited_paper_ids"])
    m = re.search(r"\n\s*Sources?:(.*)$", item["generated_answer"], re.DOTALL)
    block = m.group(1) if m else ""
    return {pid for title, pid in title_to_pid.items() if title in block}


def main(results_path):
    data = json.loads((common.ROOT / results_path).read_text())
    title_to_pid = {p["title"]: p["paper_id"] for p in common.load_papers()}
    manual = json.loads(MANUAL_PATH.read_text()) if MANUAL_PATH.exists() else {}

    rows, missing = [], []
    for item in data:
        verdict = item.get("objective_grounding", {}).get("verdict")
        cited = cited_paper_ids(item, title_to_pid)
        citation_ok = cited == set(item["gold_paper_ids"])
        auto = (verdict == "pass" and citation_ok) if item["case"] != 6 else None
        judgment = manual.get(item["id"])
        if judgment is not None:
            correct, source = judgment["correct"], "manual"
        elif auto is not None:
            correct, source = auto, "auto"
        else:
            correct, source = None, "missing"
            missing.append(item["id"])
        rows.append({
            "id": item["id"], "case": item["case"], "grounding": verdict,
            "cited_paper_ids": sorted(cited), "citation_ok": citation_ok,
            "auto_correct": auto, "correct": correct, "source": source,
            "rationale": judgment.get("rationale") if judgment else None,
        })

    def tally(subset):
        judged = [r for r in subset if r["correct"] is not None]
        return sum(r["correct"] for r in judged), len(subset)

    print(f"{'Case':<12}{'Correct':<10}{'Items'}")
    summary = {}
    for case in [1, 2, 3, 4, 5, 6]:
        subset = [r for r in rows if r["case"] == case]
        if not subset:
            continue
        c, n = tally(subset)
        summary[str(case)] = {"correct": c, "n": n}
        ids = ", ".join(r["id"] + ("*" if r["source"] == "manual" else "") for r in subset if r["correct"])
        print(f"{case:<12}{f'{c}/{n}':<10}{ids}")
    for label, subset in [("cases_1_4", [r for r in rows if r["case"] in (1, 2, 3, 4)]), ("all", rows)]:
        c, n = tally(subset)
        summary[label] = {"correct": c, "n": n}
        print(f"{label:<12}{f'{c}/{n}':<10}{c / n:.2%}")
    print("(* = decided by data/manual_judgments.json)")
    if missing:
        print(f"WARNING: no manual judgment for Case 6 items: {missing}")

    out_path = common.summary_path_for(common.ROOT / results_path).with_name(
        (common.ROOT / results_path).stem + ".answer_correct.json")
    out_path.write_text(json.dumps({"summary": summary, "items": rows}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/results.json")

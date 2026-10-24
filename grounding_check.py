"""Objective, reproducible Answer Grounding check -- replaces subjective
eyeballing with a documented rubric anyone can re-run: each QA item lists
explicit `grounding_facts` (numbers with tolerance, or key terms with
aliases) extracted directly from its verified reference_answer, and this
module checks which of them literally appear in a generated answer.

This checks *fact coverage*, not full semantic correctness -- e.g. it cannot
detect that a number is present but wrongly attributed to the wrong cause
(see q05 in the pilot). That limitation is deliberate and disclosed rather
than papered over: a coverage check is necessarily a lower bound on
correctness, not a substitute for reading the answer.
"""
import re

NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def _numbers_in(text):
    return [float(m) for m in NUMBER_RE.findall(text)]


def check_fact(generated_answer, fact, tol=0.15):
    if fact["type"] == "number":
        target = fact["value"]
        return any(abs(n - target) <= tol for n in _numbers_in(generated_answer))
    # text fact: case-insensitive substring match against the value or any alias
    text = generated_answer.lower()
    variants = [fact["value"]] + fact.get("aliases", [])
    return any(v.lower() in text for v in variants)


def score_grounding(generated_answer, grounding_facts):
    results = [
        {"fact": f["value"], "type": f["type"], "matched": check_fact(generated_answer, f)}
        for f in grounding_facts
    ]
    n_matched = sum(r["matched"] for r in results)
    coverage = n_matched / len(results) if results else 0.0
    verdict = "pass" if coverage == 1.0 else ("fail" if coverage == 0.0 else "partial")
    return {"coverage": coverage, "verdict": verdict, "facts": results}


if __name__ == "__main__":
    import json
    import sys
    import common

    results_path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
    qa_name = sys.argv[2] if len(sys.argv) > 2 else "data/qa_set.json"
    qa_by_id = {q["id"]: q for q in common.load_qa(qa_name)}
    data = json.loads((common.ROOT / results_path).read_text())

    tally = {"pass": 0, "partial": 0, "fail": 0}
    tally_by_case = {}
    for item in data:
        qa = qa_by_id[item["id"]]
        grading = score_grounding(item["generated_answer"], qa["grounding_facts"])
        item["objective_grounding"] = grading
        tally[grading["verdict"]] += 1
        case_key = str(item["case"])
        tally_by_case.setdefault(case_key, {"pass": 0, "partial": 0, "fail": 0})
        tally_by_case[case_key][grading["verdict"]] += 1
        missing = [f["fact"] for f in grading["facts"] if not f["matched"]]
        print(f"{item['id']} ({qa['case_label']}): {grading['verdict']} "
              f"({grading['coverage']:.0%} coverage)"
              + (f" -- missing: {missing}" if missing else ""))

    print(f"\nTotal: {tally}")
    (common.ROOT / results_path).write_text(json.dumps(data, indent=2))
    print(f"Wrote objective_grounding into {results_path}")

    summary_path = common.summary_path_for(common.ROOT / results_path)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"by_case": {}, "all": {}}
    for case_key, t in tally_by_case.items():
        summary["by_case"].setdefault(case_key, {})["objective_grounding"] = t
    summary["all"]["objective_grounding"] = tally
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote objective grounding tally into {summary_path}")

"""Backfills a <results>.summary.json from an already-saved results file,
without re-running the retrieval/generation pipeline. Useful for results
files produced before summary-writing was added to evaluate.py, or to
regenerate a summary after re-running grounding_check.py.
"""
import json
import sys

import common

METRICS = ["paper_recall_at_1", "paper_recall_at_5", "evidence_recall_at_1", "evidence_recall_at_5"]


def build_summary(results):
    summary = {"by_case": {}, "all": {}}
    cases = sorted({str(r["case"]) for r in results}, key=lambda c: (c.isdigit() is False, c))
    for case in cases:
        rows = [r for r in results if str(r["case"]) == case]
        summary["by_case"][case] = {"n": len(rows), **{m: common.avg(rows, m) for m in METRICS}}
        og = [r["objective_grounding"]["verdict"] for r in rows if "objective_grounding" in r]
        if og:
            summary["by_case"][case]["objective_grounding"] = {
                v: og.count(v) for v in ("pass", "partial", "fail")
            }
    summary["all"] = {"n": len(results), **{m: common.avg(results, m) for m in METRICS}}
    og_all = [r["objective_grounding"]["verdict"] for r in results if "objective_grounding" in r]
    if og_all:
        summary["all"]["objective_grounding"] = {v: og_all.count(v) for v in ("pass", "partial", "fail")}
    return summary


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
    results = json.loads((common.ROOT / results_path).read_text())
    summary = build_summary(results)
    out_path = common.summary_path_for(common.ROOT / results_path)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))

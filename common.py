"""Shared paths, corpus loading, chunking, and CSV-to-markdown helpers for the pilot."""
import csv
import difflib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "index"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PAPERS_JSON = DATA_DIR / "papers.json"

HEADER_RE = re.compile(r"^(\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 ,'\-]{2,80}|Abstract|References)$")


def load_papers():
    return json.loads(PAPERS_JSON.read_text())


def load_paper_meta():
    """paper_id -> {title, authors, conf, nicknames} for citing lookup-able info
    in generated answers, not just an internal artifact id."""
    return {p["paper_id"]: p for p in load_papers()}


def avg(rows, key):
    """Mean of rows[key], skipping None (ambiguous-item placeholders). None if empty."""
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def summary_path_for(results_path):
    """results_v13_qav4.json -> results_v13_qav4.summary.json"""
    p = Path(results_path)
    return p.with_suffix("").with_suffix(".summary.json") if p.suffix == ".json" else Path(str(p) + ".summary.json")


def format_citation(meta):
    """{'title':..., 'authors':..., 'conf': 'ACL2025'} -> a human, lookup-able
    citation string: 'Title -- Authors (Venue Year)'."""
    m = re.match(r"([A-Za-z]+)(\d{4})", meta.get("conf", ""))
    venue_year = f"{m.group(1)} {m.group(2)}" if m else meta.get("conf", "unknown venue")
    return f"\"{meta['title']}\" -- {meta['authors']} ({venue_year})"


def load_qa(qa_name="data/qa_set.json"):
    return json.loads((ROOT / qa_name).read_text())


def paper_paths(paper_id):
    d = CORPUS_DIR / paper_id
    return {
        "dir": d,
        "text": d / "full.txt",
        "tables_dir": d / "tables",
        "figures_dir": d / "figures",
    }


def chunk_full_text(text, target_chars=800):
    """Split into paragraph-merged chunks, tagging each with the nearest preceding
    section header. Returns a list of {chunk_text, header} dicts."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_header = "(preamble)"
    buf = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append({"chunk_text": "\n\n".join(buf), "header": current_header})
            buf, buf_len = [], 0

    for para in paragraphs:
        first_line = para.splitlines()[0].strip()
        if HEADER_RE.match(first_line) and len(para.splitlines()) == 1:
            flush()
            current_header = first_line
            continue
        if buf_len + len(para) > target_chars and buf:
            flush()
        buf.append(para)
        buf_len += len(para)
    flush()
    return chunks


def _longest_common_substring(a, b):
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    match = m.find_longest_match(0, len(a), 0, len(b))
    return a[match.a: match.a + match.size]


def _repair_flattened_header(header):
    """Docling sometimes flattens a multi-row header by concatenating the FULL
    list of leaf column names into every group cell (joined onto a per-column
    prefix/suffix with no space, e.g. 'In-domain.FF-TQA.FeTaQA TabFact WTQ
    ...FinQA.Proprietary Models'), instead of splitting one leaf name per
    column. Detect the shared middle text across cells via longest-common-
    substring, and if splitting it by whitespace yields exactly one leaf name
    per data column, reassign each column its own positional leaf name
    (prefixed by whatever unique text surrounds the shared blob)."""
    candidates = [c for c in header if len(c.split()) > 3]
    if len(candidates) < max(2, len(header) - 2):
        return header
    shared = candidates[0]
    for c in candidates[1:]:
        shared = _longest_common_substring(shared, c)
        if len(shared.split()) < 2:
            return header
    # A glued-on group label (e.g. '...FinQA.Proprietary Models') can inflate
    # the shared match by a token or two at either edge; search small trims
    # for an exact one-leaf-per-data-column split before giving up.
    tokens = shared.split()
    n_cols = len(header) - 1
    leaves = trimmed = None
    for left in range(0, min(3, len(tokens)) + 1):
        for right in range(0, min(3, len(tokens) - left) + 1):
            cand = tokens[left: len(tokens) - right] if right else tokens[left:]
            if len(cand) == n_cols:
                leaves, trimmed = cand, (left, right)
                break
        if leaves:
            break
    if leaves is None:
        return header  # not a clean one-leaf-per-data-column match; leave as-is
    left, right = trimmed
    core = " ".join(tokens[left: len(tokens) - right] if right else tokens[left:])
    new_header = [header[0]]
    for i, cell in enumerate(header[1:]):
        prefix = cell.replace(core, "\x00", 1).split("\x00")[0].strip(" .")
        leaf = leaves[i].strip(" .")
        new_header.append(f"{prefix} / {leaf}" if prefix else leaf)
    return new_header


def _flag_ambiguous_symbol_columns(header, body):
    """Columns whose values are all short non-numeric symbols (e.g. Docling's
    mangled checkmarks '!'/'%') have no self-evident meaning. Flag them in a
    trailing note instead of silently feeding a guessable-wrong label."""
    flagged = []
    for j, name in enumerate(header):
        vals = {r[j].strip() for r in body if j < len(r) and r[j].strip()}
        if vals and all(len(v) <= 2 and not re.match(r"^-?\d+(\.\d+)?%?$", v) for v in vals):
            flagged.append((name, sorted(vals)))
    return flagged


def csv_to_markdown(csv_path, max_rows=40):
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    rows = rows[:max_rows]
    header, body = rows[0], rows[1:]
    header = _repair_flattened_header(header)
    out = io.StringIO()
    out.write("| " + " | ".join(header) + " |\n")
    out.write("|" + "|".join(["---"] * len(header)) + "|\n")
    for r in body:
        r = r + [""] * (len(header) - len(r))
        out.write("| " + " | ".join(r[: len(header)]) + " |\n")
    flagged = _flag_ambiguous_symbol_columns(header, body)
    if flagged:
        out.write("\nNote: the following column(s) use symbols whose on/off meaning is "
                   "not specified by the extracted table -- interpret with caution "
                   "rather than assuming which symbol means enabled:\n")
        for name, vals in flagged:
            out.write(f"- '{name}': symbols used = {vals}\n")
    return out.getvalue()


def extract_title_abstract(full_text):
    """First non-empty line as title, plus the text between the 'Abstract' line
    and the next section header (e.g. '1 Introduction')."""
    lines = [l.strip() for l in full_text.splitlines()]
    title = next((l for l in lines if l), "")
    if "Abstract" not in lines:
        return title, ""
    start = lines.index("Abstract") + 1
    body = []
    for l in lines[start:]:
        if l and HEADER_RE.match(l):
            break
        body.append(l)
    return title, " ".join(l for l in body if l).strip()


def find_caption(full_text, kind, number):
    """kind: 'Table' or 'Figure'. number: int parsed from filename."""
    pat = re.compile(rf"^{kind} {number}:\s*(.+)$", re.MULTILINE)
    m = pat.search(full_text)
    return m.group(0).strip() if m else f"{kind} {number} (caption not found)"


def artifact_id(paper_id, kind, number_str):
    return f"{paper_id}#{kind}_{number_str}"

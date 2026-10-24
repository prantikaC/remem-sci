# ReMem-Sci: a memory-conditioned multimodal RAG pilot for scholarly retrieval

Researchers rarely remember a paper cleanly. They remember *that Yale table-reasoning paper*, *a scatter
plot somewhere between 5 and 25 seconds*, or a number that is slightly wrong. This repository is a small,
fully local pilot that tests a multimodal retrieval-augmented generation (RAG) system against those kinds
of partial memories. It covers 15 real ACL/EMNLP 2025 papers and 24 hand-written questions, spread across
six memory cases.

The retrieval target is an **evidence fragment** (a passage, table or figure), not just a paper. Each
answer is scored on three separate things: did the system find the right paper, did it find the right
evidence, and is the final answer correct and correctly cited.

---

## Contents

- [The six memory cases](#the-six-memory-cases)
- [Repository layout](#repository-layout)
- [Corpus: the 15 papers](#corpus-the-15-papers)
- [QA dataset](#qa-dataset)
- [System](#system)
- [Evaluation](#evaluation)
- [Results](#results)
- [Results by case, with examples](#results-by-case-with-examples)
- [Ablation: how papers are ranked](#ablation-how-papers-are-ranked)
- [Reproducing](#reproducing)
- [Known limitations](#known-limitations)

---

## The six memory cases

Each case is a different observability condition on the researcher's memory of the paper (**P**) and of
the evidence inside it (**E**). The case label is evaluation metadata only. The system never sees it, and
one pipeline answers every question.

| Case | Name | What the researcher remembers |
|---|---|---|
| 1 | Evidence-Specific Memory | Paper and evidence both reasonably well known (the paper is referred to by nickname, lab or method, never by exact title) |
| 2 | Evidence-Uncertain Memory | The paper is known; where the evidence sits in it is forgotten |
| 3 | Paper-Uncertain Memory | The paper is not identified, but the evidence is described distinctively |
| 4 | Weak Joint Memory | Both the paper and the evidence are vague |
| 5 | Cross-Paper Memory | The recollection spans several papers (a comparison, a relationship between approaches, or facts from different papers); the system must find each paper and attribute each fact to the right one |
| 6 | Inaccurate Memory | A real, identifiable paper plus one confidently stated detail that is wrong; the right behaviour is to catch the false premise, not confirm it |

---

## Repository layout

```
.
├── data/
│   ├── qa_set.json              # the 24-item QA dataset (4 per case)
│   ├── papers.json              # paper manifest: id, title, authors, venue, nicknames
│   └── manual_judgments.json    # human judgments used by answer_correct.py, with a reason for each
├── corpus/<paper_id>/
│   ├── full.txt                 # extracted running text
│   ├── tables/table_NNN.csv     # extracted tables
│   └── figures/figure_NNN.png   # extracted figures
├── index/
│   ├── catalog.json             # all 1,922 evidence artifacts (text, captions, figure descriptions)
│   ├── artifacts.faiss          # dense index over the catalog
│   ├── figure_caption_cache.json
│   └── paper_level_embeddings.npz   # used only by the ranking ablation
├── results/
│   ├── results_final.json                 # per-question retrieval, generated answer, grounding
│   ├── results_final.summary.json         # recall and grounding per case
│   ├── results_final.answer_correct.json  # Answer Correct per question and per case
│   └── paper_ranking_ablation_results.json
├── common.py                    # paths, chunking, CSV->markdown, citation formatting
├── vlm_utils.py                 # Qwen2.5-VL-7B-Instruct loader / generate()
├── build_index.py               # corpus -> catalog + FAISS index
├── answer.py                    # the retrieval + grounded generation pipeline
├── evaluate.py                  # runs the QA set, computes paper/evidence recall
├── grounding_check.py           # objective fact-coverage grounding check
├── answer_correct.py            # final Answer Correct metric
├── make_summary.py              # rebuilds a summary file from a results file
├── paper_level_embeddings.py    # builds abstract / mean-pool paper embeddings (ablation)
└── paper_ranking_ablation.py    # compares paper-ranking strategies (ablation)
```

---

## Corpus: the 15 papers

All 15 papers are from the ACL Anthology: ACL 2025 (long papers) and EMNLP 2025 (main conference). They
were chosen to be topically close (RAG, retrieval, multimodal and table QA, benchmarks), so that confusing
one paper for another is realistic. Each PDF was converted with
[Docling](https://github.com/DS4SD/docling) into running text, table CSVs and figure images.

| # | Paper | Authors (affiliation) | Venue | Type | Artifacts (P / T / F) | Gold for |
|---|---|---|---|---|---|---|
| 1 | [Table-R1: Inference-Time Scaling for Table Reasoning](https://aclanthology.org/2025.emnlp-main.1040/) | Zheyuan Yang, Lyuhao Chen, Arman Cohan, Yilun Zhao (Yale NLP Lab / CMU) | EMNLP 2025 | Method: RL / distillation post-training for table reasoning | 95 / 2 / 5 | q44 (C1), q01 (C4), q29 (C5) |
| 2 | [REARANK: Reasoning Re-ranking Agent via Reinforcement Learning](https://aclanthology.org/2025.emnlp-main.125/) | Le Zhang, Bo Wang, Xipeng Qiu, Siva Reddy, Aishwarya Agrawal (Mila / Fudan / McGill) | EMNLP 2025 | Method: RL-trained listwise reranking agent | 65 / 5 / 5 | q45 (C1), q02 (C4), q29 (C5) |
| 3 | [RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](https://aclanthology.org/2025.acl-long.819/) | Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yongkang Wu, Zhonghua Li, Qi Ye, Zhicheng Dou (Renmin / Tsinghua / Huawei) | ACL 2025 | Method: retrieval inside generation via FM-Index constrained decoding | 152 / 4 / 2 | q42 (C1), q28 (C2), q03 (C4), q33 (C6) |
| 4 | [KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation](https://aclanthology.org/2025.acl-long.929/) | Jinyuan Fang, Zaiqiao Meng (University of Glasgow) | ACL 2025 | Method: iterative retrieval over knowledge triples | 103 / 16 / 11 | q04 (C2), q25 (C3), q16 (C4), q34 (C6) |
| 5 | [OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval](https://aclanthology.org/2025.acl-long.1198/) | Wei Yang, Jingjing Fu, Rui Wang, Jinyu Wang, Lei Song, Jiang Bian (Microsoft) | ACL 2025 | Method: coarse-to-fine multimodal retrieval for knowledge-based VQA | 111 / 5 / 1 | q27 (C1), q05 (C2), q14 (C5), q35 (C6) |
| 6 | [How to Train Long-Context Language Models (Effectively)](https://aclanthology.org/2025.acl-long.366/) | Tianyu Gao, Alexander Wettig, Howard Yen, Danqi Chen (Princeton) | ACL 2025 | Empirical study: long-context training recipe (ProLong) | 142 / 8 / 2 | distractor |
| 7 | [COIR: A Comprehensive Benchmark for Code Information Retrieval Models](https://aclanthology.org/2025.acl-long.1072/) | Xiangyang Li, Kuicai Dong, Yi Quan Lee, et al. (Huawei Noah's Ark Lab) | ACL 2025 | Benchmark: code retrieval | 97 / 6 / 2 | q24 (C3), q15 (C5) |
| 8 | [REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](https://aclanthology.org/2025.acl-long.1528/) | Navve Wasserman, Roi Pony, Oshri Naparstek, et al. (IBM Research Israel) | ACL 2025 | Benchmark: multimodal document retrieval | 106 / 2 / 3 | q14 (C5) |
| 9 | [DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics](https://aclanthology.org/2025.acl-long.1127/) | Yayu Long, Kewei Chen, Long Jin, Mingsheng Shang (Chinese Academy of Sciences) | ACL 2025 | Method: mixture-of-experts + RAG for robotics | 214 / 17 / 2 | distractor |
| 10 | [NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](https://aclanthology.org/2025.acl-long.311/) | Ruisheng Cao, Hanchong Zhang, Tiancheng Huang, et al. (Shanghai Jiao Tong University) | ACL 2025 | Method: relational DB + vector store retrieval for PDF QA | 117 / 2 / 10 | q10 (C3), q36 (C6) |
| 11 | [Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering](https://aclanthology.org/2025.acl-long.1089/) | Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei Wang, Wei Hu (Nanjing University) | ACL 2025 | Method: entity-aware multi-hop retrieval (ChainRAG) | 94 / 3 / 5 | q13 (C5) |
| 12 | [MS-RAG: Simple and Effective Multi-Semantic Retrieval-Augmented Generation](https://aclanthology.org/2025.emnlp-main.1151/) | Xiaozhou You, Yahui Luo (Ant Group) | EMNLP 2025 | Method: knowledge graph + dense multi-level indexing | 91 / 9 / 2 | q47 (C2) |
| 13 | [M3GQA: A Multi-Entity Multi-Hop Multi-Setting Graph Question Answering Benchmark](https://aclanthology.org/2025.acl-long.1478/) | Boci Peng, Yongchao Liu, Xiaohe Bo, et al. (Peking University / Ant Group) | ACL 2025 | Benchmark: GraphRAG, multi-entity questions | 147 / 5 / 2 | q15 (C5) |
| 14 | [Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models](https://aclanthology.org/2025.emnlp-main.1063/) | Qiang Liu, Xinlong Chen, Yue Ding, Bowen Song, et al. (CASIA / Ant Group) | EMNLP 2025 | Method: zero-shot hallucination detection (AGSER) | 102 / 11 / 2 | distractor |
| 15 | [DualRAG: A Dual-Process Approach to Integrate Reasoning and Retrieval for Multi-Hop Question Answering](https://aclanthology.org/2025.acl-long.1539/) | Rong Cheng, Jinyi Liu, Yan Zheng, et al. (Tianjin University) | ACL 2025 | Method: coupled reasoning + retrieval for multi-hop QA | 130 / 5 / 2 | q17 (C3), q13 (C5) |

P / T / F = passages / tables / figures in the index (1,766 / 100 / 56; 1,922 in total). "Distractor"
papers have no question pointed at them, but they are indexed and compete in every search.

**License of the corpus content.** ACL Anthology papers are distributed under the
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) license. The extracted
text, tables and figures in `corpus/` and `index/` are redistributed here with attribution to the papers
above.

---

## QA dataset

`data/qa_set.json` contains 24 hand-written questions, 4 per case. Every answer was checked by hand
against the paper's actual text, table or figure. The questions are written the way a researcher types
into a chat box: informal, hedged, and referring to papers by nickname or description, never by title.

| Field | Meaning |
|---|---|
| `id`, `case`, `case_label` | Identifier and memory case (1–6) |
| `query_text` | The question as the researcher would ask it |
| `gold_paper_id` / `gold_paper_ids` | The correct paper (one, or several for Case 5) |
| `gold_artifact_id` | The exact gold evidence artifact, when there is a single one (13 questions) |
| `gold_keywords` | Otherwise: keywords that a gold-paper passage must contain to count as the evidence (7 questions) |
| `gold_keywords_by_paper` | Case 5: keywords per gold paper; each paper must independently surface matching evidence (4 questions) |
| `gold_modality` | Modality of the gold evidence: passage (11), table (10), figure (3) |
| `reference_answer` | The verified answer, with a full citation and its location (table / figure / section) |
| `grounding_facts` | The numbers (with tolerance) or key terms (with aliases) a correct answer must contain |

---

## System

One pipeline answers every question. It works paper-first and in two stages, so paper identification and
evidence localization are measured separately.

**Index (`build_index.py`).** Every paper becomes a set of evidence artifacts:
- **Passages**: paragraph-merged chunks of about 800 characters, each tagged with its nearest section header.
- **Tables**: the paper's real caption (taken from the text) plus the table rendered as markdown.
- **Figures**: the real caption plus a short visual description written by Qwen2.5-VL-7B-Instruct
  (chart type, trend, layout), so a figure can be found from a visual memory.

All artifacts are embedded with `BAAI/bge-large-en-v1.5` into a single FAISS inner-product index.

**Query time (`answer.py`).**
1. **Paper identification.** The query is embedded and scored against *every* artifact in the corpus.
   Each paper's score is its best-scoring artifact. The top 5 papers form a shortlist.
2. **Evidence localization.** Only artifacts from the shortlisted papers are kept (top 30). The
   `BAAI/bge-reranker-base` cross-encoder reranks them down to 5.
3. **Grounded generation.** The 5 artifacts go to `Qwen/Qwen2.5-VL-7B-Instruct`, along with the images of
   any figures among them. Decoding is greedy, with at most 400 new tokens. The model answers from the
   evidence only, is asked to mention the table, figure or section where the answer is when that is
   available, and ends with the numbers of the evidence items it used.
4. **Citation.** The title, authors and venue of each cited paper are attached from `papers.json`,
   never from the model's own memory.

Hardware: one NVIDIA L40S GPU (48 GB) per run, on a 3× L40S machine. Everything runs locally; no
external API is used.

---

## Evaluation

| Metric | Definition |
|---|---|
| **Paper R@1** | The top-ranked paper is a gold paper |
| **Paper R@5** | All gold papers are among the top 5 ranked papers |
| **Evidence R@1 / R@5** | The gold evidence is among the top 1 / 5 reranked artifacts: an exact `gold_artifact_id` match, or a gold-paper passage containing a `gold_keywords` term, or (Case 5) matching evidence from *every* gold paper |
| **Objective grounding** | The share of `grounding_facts` found in the generated answer: **pass** = all, **partial** = some, **fail** = none |
| **Answer Correct** | Cases 1–5: grounding is **pass** (partial counts as incorrect) **and** the cited papers are exactly the gold papers. Case 6: correct if the answer says the claim is not supported by the evidence, or explicitly corrects it with accurate information; incorrect if it confirms the false premise |

Answer Correct is computed by `answer_correct.py`. Case 6 needs a person to read the answer, and so does
any question where the automatic rule is misleading. Those decisions are in `data/manual_judgments.json`,
each with its reason, so every number in the results can be traced.

Recall is not reported for Cases 5 and 6. Case 5 needs joint coverage of several papers rather than one
gold paper, and Case 6 is about catching the false premise, not about recall.

---

## Results

`results/results_final.json`

| Case | N | Paper R@1 | Paper R@5 | Evidence R@1 | Evidence R@5 | Answer Correct |
|---|---|---|---|---|---|---|
| 1 | 4 | 100% | 100% | 50% | 75% | 1/4 (25%) |
| 2 | 4 | 100% | 100% | 50% | 75% | 1/4 (25%) |
| 3 | 4 | 75% | 100% | 50% | 100% | 2/4 (50%) |
| 4 | 4 | 75% | 100% | 25% | 50% | 1/4 (25%) |
| 5 | 4 | multi-paper evaluation | | | | 1/4 (25%) |
| 6 | 4 | recall not applicable | | | | 1/4 (25%) |
| **Cases 1–4** | 16 | **87.5%** | **100%** | **43.75%** | **75%** | **5/16 (31.25%)** |
| **All** | 24 | | | | | **7/24 (29.17%)** |

Objective grounding (pass / partial / fail): Case 1: 1/0/3 · Case 2: 1/0/3 · Case 3: 2/1/1 ·
Case 4: 1/1/2 · Case 5: 2/2/0 · Case 6: 0/3/1 · total 7/7/10.

For reference, multi-paper retrieval for Case 5 reached Paper R@5 = 100% and Evidence R@5 = 50%.

**Main finding.** Finding a plausible paper is much easier than finding the specific evidence: Paper R@5
is 100% while Evidence R@5 is 75% (Cases 1–4). Finding the evidence is in turn easier than producing a
correct, correctly cited answer: Answer Correct is 31.25%. Paper identification, evidence localization
and answer grounding behave as three separate capabilities.

**Provenance.** All answers were generated on 2026-09-11. Cases 1–4 and 6 come from the full 24-question
run. Case 5 was rewritten afterwards so that it matches the Cross-Paper definition, and its four questions
were then run again with the same pipeline. Each question's `run` field records which run it comes from.
Questions are scored independently, so combining the two runs is valid.

---

## Results by case, with examples

Each example shows the question, the system's answer (shortened where marked "…"), the reference answer,
and why the answer was judged correct or incorrect. The full texts are in `results/results_final.json`.

### Case 1: Evidence-Specific Memory

Paper R@1 is 100%, and the exact evidence is found in 3 of 4 questions. Yet only 1 of 4 answers is correct.
When memory is strong, retrieval mostly succeeds, and the bottleneck moves to generation.

**Success: `q42`.**
> **Q:** *I remember the FM-Index retrieval-generation paper had a table comparing storage and build time against a standard dense-embedding approach -- theirs came out to roughly a third the size and noticeably quicker to build. What were the exact numbers?*
>
> **System:** "…the FM-Index method resulted in 20.7 GB of total storage and 5.39 hours for building time, compared to 60.1 GB and 7.57 hours for the E5-Base method…" *(Source: RetroLLM, ACL 2025)*
>
> **Reference:** RetroLLM's FM-Index takes 20.7 GB and 5.39h to build, vs. E5-Base's dense embeddings at 60.1 GB and 7.57h (RetroLLM, Table 9).

The paper and the exact table were both ranked first, and all four numbers are correct.

**Failure: `q44`.**
> **Q:** *Pretty sure it was that Yale table-reasoning paper where their RL-trained version beat the plain baseline by around 10 points on one of the table QA benchmarks. Which one, and what were the exact scores?*
>
> **System:** "…the table in Evidence 1 shows that the "Reasoning Language Model" achieved an EM accuracy of approximately 33.8% on the TQA task, which is significantly higher than the baseline model's score…" *(Source: Table-R1, EMNLP 2025)*
>
> **Reference:** On FeTaQA, Table-R1-Zero scores 32.7 vs. 21.7 for the plain Llama-3.1-8B-Instruct baseline (Table-R1, Table 2).

Retrieval was perfect here: the right paper and the exact gold table were both ranked first. The model
still invented a model name and a score (33.8%), and pointed to the wrong task instead of FeTaQA. This
is a pure generation failure.

### Case 2: Evidence-Uncertain Memory

The paper is always found (Paper R@1 100%), but the right evidence inside it is found in 3 of 4 questions
by rank 5 and only 2 of 4 at rank 1. Knowing the paper does not tell the system where in it to look.

**Success: `q28`.**
> **Q:** *Can you remind me what that FM-Index retrieval-generation paper actually did about the failure mode they found, something about pruning too aggressively during constrained decoding? They built two separate fixes around it -- I just don't remember which section covered it.*
>
> **System:** "The FM-Index retrieval-generation paper discusses a severe false pruning problem in constrained decoding… The paper proposes two fixes: hierarchical FM-Index constraints and forward-looking constrained decoding…" *(Source: RetroLLM, ACL 2025)*
>
> **Reference:** Two fixes: hierarchical FM-Index constraints, which narrow decoding to a relevant subset of documents before evidence generation, and forward-looking constrained decoding, which considers the relevance of future sequences (RetroLLM).

The right passage was not ranked first, but it was in the top 5, and the model picked it out and named
both fixes.

**Failure: `q05`.**
> **Q:** *There was a paper about answering questions about pictures using outside information, searching in more than one round -- first broadly then more precisely -- from a large tech company. … I think one of the middle rounds mattered more than the others. Which one, and by how much?*
>
> **System:** "The evidence provided does not directly address the specific question about the importance of a particular round… Therefore, no single evidence item answers the question." *(Source: OMGM, ACL 2025)*
>
> **Reference:** The multimodal fusion reranking stage (step 2) matters most: InfoSeek jumps from 18.87 to 31.38, while step 3 adds only ~2 points (OMGM, Table 4).

The paper was identified correctly, but the gold ablation table (Table 4) never reached the top 5. The
model declined instead of guessing, which is the right behaviour, but the question went unanswered.

### Case 3: Paper-Uncertain Memory

Paper R@1 drops to 75%, yet Evidence R@5 is 100%, the highest of any case, and so is Answer Correct
(2 of 4). A distinctive description of the content leads straight to the evidence even when the paper
itself is not ranked first.

**Success: `q17`.**
> **Q:** *There's a multi-hop QA paper where two coupled processes run side by side -- one drives the reasoning and figures out what's still missing, the other organizes everything retrieved so far into a running knowledge outline that feeds back into the first. … How do the two pieces actually work together?*
>
> **System:** "The DualRAG framework… consists of two coupled processes: Reasoning-augmented Querying (RaQ) and progressive Knowledge Aggregation (pKA)… pKA organizes the retrieved information into a running knowledge outline, which is then fed back into RaQ… enables DualRAG to approach or even surpass oracle-level retrieval performance." *(Source: DualRAG, ACL 2025)*
>
> **Reference:** DualRAG; its two processes are Reasoning-augmented Querying (RaQ) and progressive Knowledge Aggregation (pKA), and it approaches or surpasses oracle performance.

DualRAG was *not* the top-ranked paper (Paper R@1 failed), but the description was distinctive enough for
the right passage to reach the top 5. The answer names both processes and is cited correctly.

**Failure: `q10`.**
> **Q:** *Been trying to place a paper about answering questions by reading whole documents, where they stored parsed information in more than one kind of database at once, structured and unstructured. There was some diagram of this setup -- what did it actually show?*
>
> **System:** "…storing parsed information in multiple databases simultaneously… The diagram in Evidence 3 illustrates the structure of the database used to store parsed elements of each PDF file, including tables like 'images', 'equations', 'pages', 'chunks'…" *(Source: NeuSym-RAG, ACL 2025)*
>
> **Reference:** NeuSym-RAG organizes parsed PDF content into both a relational database and a vector store; Figure 10 shows the database schema.

The paper and the schema evidence were both found, but the answer mostly repeats the question's own
wording ("multiple databases") and never says *which* two stores are used: a relational database and a
vector store. That missing fact is exactly the one the researcher could not recall.

### Case 4: Weak Joint Memory

This is the hardest of the single-paper cases. It has the lowest evidence recall (25% at rank 1, 50% at
rank 5), and only 1 of 4 answers is correct. When both the paper and the evidence are vague, the system
can confidently pick the wrong paper.

**Success: `q16`.**
> **Q:** *I remember a scatter plot from some paper on retrieval -- a bunch of colored dots trading off accuracy against how long each query took, roughly clustered somewhere between 5 and 25 seconds on the x-axis and up in the 70s-to-90-percent range on the y-axis. No memory of the method names or which paper it was from, just that plot. Can you figure out which paper that's from?*
>
> **System:** "The scatter plot described in Evidence 1 matches the researcher's recollection. It shows a trade-off between latency and retrieval accuracy, with models like KiRAG (online) achieving high R@3 percentages (around 70-90%) while latency ranges from about 5 to 25 seconds…" *(Source: KiRAG, ACL 2025)*
>
> **Reference:** Figure 5 of KiRAG: R@3 vs. latency, roughly 5–25 s and 70–90%.

A purely visual memory was enough. The figure's machine-written visual description was indexed, so the
exact figure was ranked first.

**Failure: `q01`.**
> **Q:** *I remember reading a paper that trained a model to answer questions about spreadsheet-like data, using trial-and-reward style training instead of plain supervised learning. There was a table somewhere showing it doing noticeably better than the untrained starting point on one of the tasks. Can you tell me roughly how much better it did?*
>
> **System:** "…based on the available evidence, we cannot determine how much better the model performed." *(Source: REAL-MM-RAG, ACL 2025)*
>
> **Reference:** About 11 points: Table-R1-Zero scores 32.7 on FeTaQA vs. 21.7 for the base model (Table-R1, Table 2).

The vague description ("spreadsheet-like data", "trial-and-reward") did not put Table-R1 first, the gold
table was not retrieved, and the answer cites the wrong paper.

### Case 5: Cross-Paper Memory

Every gold paper is found (Paper R@5 = 100%), and evidence from both papers is jointly in the top 5 for
half the questions (Evidence R@5 = 50%). Only 1 of 4 answers is correct. The bottleneck is
**attribution**: the model often reasons correctly about both papers but cites only one.

**Success: `q15`.**
> **Q:** *Two new benchmark papers from this year that I keep mixing up -- one breaks its evaluation into four main tasks with several sub-tasks under each, for one kind of retrieval; the other defines six different settings, including one about editing the answer and one about checking whether an answer even exists in a smaller graph. Which is the code one and which is the graph one, and how many tasks or settings does each actually define?*
>
> **System:** "The first paper… describes a benchmark with six different settings… This aligns with the graph retrieval task. The second paper… evaluates retrieval models on eight subtasks, suggesting it focuses on code retrieval tasks…" *(Sources: M3GQA; COIR, both ACL 2025)*
>
> **Reference:** COIR is the code benchmark (4 primary tasks, 8 sub-tasks, 10 datasets); M3GQA is the graph benchmark (six settings).

Both papers are told apart correctly, and both are cited. The answer leaves out COIR's count of four main
tasks rather than guessing it.

**Failure: `q29`.**
> **Q:** *Both of these papers train their model with the same RL algorithm, GRPO -- one for reranking search results, the other for table reasoning. But their reward signals aren't measuring the same thing: one's reward is based on ranking quality, the other's is based on getting the exact answer right. Which is which?*
>
> **System:** "…the model trained for reranking search results uses a reward signal based on ranking quality, specifically measured by NDCG@10… the model trained for table reasoning uses accuracy rewards, which measure the correctness of the model's outputs…" *(Source: REARANK only)*
>
> **Reference:** REARANK's reward is NDCG@10-based; Table-R1's reward is answer correctness (e.g. exact match). Both use GRPO.

The reasoning is right for *both* papers, but the citation names only REARANK. Table-R1 is never
attributed, so the answer fails on incomplete sourcing, not on content. `q13` fails the same way. `q14`
cites both papers but calls the benchmark "ViDoRe" instead of REAL-MM-RAG (see the known limitations).

### Case 6: Inaccurate Memory

Only 1 of 4 answers catches the false premise. The other three go along with what the researcher
confidently stated, even when the retrieved evidence contradicts it.

**Success: `q35`.**
> **Q:** *That Microsoft coarse-to-fine multimodal RAG paper -- I recall their ablation showing that combining image and text actually made retrieval worse than using text alone, which really surprised me. Is that right?*
>
> **System:** "The evidence does not directly support the claim that combining image and text made retrieval worse than using text alone… the specific ablation study mentioned in the question is not supported by the provided evidence…" *(Source: OMGM, ACL 2025)*
>
> **Reference:** That's backwards: combining image and text roughly doubled R@1, from 25.8 to 51.3 (OMGM, Table 14).

The system does not confirm the false claim; it says the evidence does not support it. That counts as
correct. It does not supply the correct direction, because Table 14 was not retrieved.

**Failure: `q33`.**
> **Q:** *That FM-Index retrieval-generation paper -- pretty sure their compact index came out to around 40GB, still way smaller than the roughly 60GB dense-embedding setup they compared against, and that's what let it build so much faster. That's right, isn't it?*
>
> **System:** "The evidence indicates that the FM-Index retrieval-generation paper's compact index was indeed around 20.7 GB, which is significantly smaller than the 60.1 GB dense-embedding setup… Therefore, the claim about the FM-Index being around 40GB and building much faster is supported by the evidence." *(Source: RetroLLM, ACL 2025)*
>
> **Reference:** Not quite: the index is 20.7 GB, not ~40 GB, versus 60.1 GB (RetroLLM, Table 9).

This is the clearest failure in the set. The model retrieves and even states the true number (20.7 GB),
then declares the researcher's 40 GB claim "supported". It has the right fact and still validates the
wrong premise.

---

## Ablation: how papers are ranked

The pipeline ranks papers by their single best-matching artifact. `paper_ranking_ablation.py` compares
that with two alternatives: an embedding of each paper's title plus abstract, and the mean of all its
artifact embeddings. Only the paper-identification stage is compared; no answers are generated.

| Strategy | Cases 1–4 R@1 | Cases 1–4 R@5 | All R@1 | All R@5 |
|---|---|---|---|---|
| Best artifact (pipeline) | **87.5%** | **100%** | **87.5%** | **100%** |
| Title + abstract | 75% | 93.75% | 75% | 95.8% |
| Mean of artifacts | 87.5% | 93.75% | 87.5% | 95.8% |

Neither alternative beats the best-artifact ranking. The questions refer to specific details from the
middle of a paper, which a matching chunk captures better than an abstract or an averaged vector does.
Per-case numbers are in `results/paper_ranking_ablation_results.json`.

---

## Reproducing

```bash
pip install -r requirements.txt
# models (downloaded from Hugging Face on first use):
#   BAAI/bge-large-en-v1.5, BAAI/bge-reranker-base, Qwen/Qwen2.5-VL-7B-Instruct

python build_index.py                     # optional: index/ is already included
python evaluate.py results/my_run.json data/qa_set.json
python grounding_check.py results/my_run.json data/qa_set.json
python make_summary.py results/my_run.json
python answer_correct.py results/my_run.json

python paper_level_embeddings.py          # ablation
python paper_ranking_ablation.py data/qa_set.json
```

A full run used about 19 GB of GPU memory. To pick a GPU, set `CUDA_VISIBLE_DEVICES`; the code always
addresses `cuda:0`.

**Reproducibility note.** Retrieval is deterministic: paper and evidence recall reproduce exactly. The
answers in `results/results_final.json` were generated with a local copy of
Qwen2.5-VL-7B-Instruct whose exact revision was not recorded. A fresh download of the model can phrase
answers differently even with greedy decoding, and that can change the grounding verdicts. Treat
`results/results_final.json` as the frozen record behind the numbers above. `data/manual_judgments.json`
applies to those answers specifically. When you generate new answers, read the Case 6 answers yourself
and update the judgments before running `answer_correct.py`.

---

## Known limitations

- **Small sample.** Four questions per case: per-case percentages are illustrative, not stable estimates.
- **The grounding check reads the whole answer, including the attached citation.** A required paper name
  can therefore be "found" in the `Source:` line rather than in what the model actually wrote. This is why
  `q14` scored *pass* even though its text names the wrong benchmark ("ViDoRe"); it is overridden to
  incorrect in `data/manual_judgments.json`.
- **The grounding check matches strings, not meaning.** It cannot tell when a correct number is
  attributed to the wrong cause, and it can miss a correct paraphrase.
- **Citations depend on the model's evidence list.** Papers are cited only if the model lists their
  evidence on its final `SOURCE_EVIDENCE` line. In Case 5 it often discusses both papers but lists
  evidence from only one.
- **Table numbering from extraction.** Table files are numbered in extraction order, which does not
  always match the paper's own table numbers. For example, RetroLLM's `table_011.csv` is the paper's
  Table 9, and its caption in the catalog is wrong. The reference answers use the paper's own numbering.

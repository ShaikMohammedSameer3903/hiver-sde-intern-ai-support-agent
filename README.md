# Hiver AI Customer Support Agent

> **Assignment**: SDE Intern Take-Home — Hiver  
> **Author**: Shaik  
> **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (2.8M tweets, 108 brands)  
> **Selected Brand**: `AppleSupport` (204K tweets, 52K conversations, 87K resolution-rich replies)

---

## What This Project Does

Turns messy, real-world Twitter customer-support conversations into a **working AI support agent** with a conservative, evidence-grounded architecture that:

1. **Classifies** incoming customer messages into 10 empirically-derived intents
2. **Retrieves** the most relevant historical resolution from 45K verified cases
3. **Generates** a grounded, traceable reply (zero hallucination)
4. **Decides** whether to automate or escalate to a human — safely

## Key Evaluation Results

| Metric | Majority Baseline | TF-IDF Baseline | **Proposed System** |
|:---|:---|:---|:---|
| Intent Accuracy | 0.1350 | 0.8100 | **0.8700** |
| Macro F1 | 0.0238 | 0.8006 | **0.8683** |

| Safety Metric | Value |
|:---|:---|
| Escalation Recall | **94.59%** (catches 70/74 high-risk cases) |
| Auto-Handle Precision | **86.21%** |
| False-Auto Rate | 13.79% |
| Automation Coverage | 14.50% |
| LLM Judge Pass Rate | **88.5%** |

---

## Quick Start (< 15 minutes)

### Prerequisites

- Python 3.10+ (tested on 3.14.2)
- ~2 GB disk space (dataset + index)
- Internet connection (one-time dataset download)

### 1. Clone & Install

```bash
git clone <repo-url> && cd project
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Download dataset (automatic via kagglehub)
python -m src.data.loader          # Extract AppleSupport subset
python -c "from src.data.loader import load_brand_tweets; from src.data.conversations import reconstruct_conversations; reconstruct_conversations(load_brand_tweets())"
```

### 3. Build Retrieval Index

```bash
python -m src.retrieval.index
```

### 4. Run Interactive CLI

```bash
python -m src.pipeline
```

### 5. Run Full Evaluation

```bash
python -m evaluation.run
```

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
project/
├── src/                          # Core source code
│   ├── config.py                 # Global configuration & paths
│   ├── pipeline.py               # End-to-end pipeline + interactive CLI
│   ├── data/
│   │   ├── loader.py             # Brand extraction from raw CSV
│   │   ├── cleaner.py            # Text cleaning (preserves user voice)
│   │   └── conversations.py      # Conversation tree reconstruction
│   ├── classification/
│   │   ├── classifier.py         # Hybrid TF-IDF + Char N-gram classifier
│   │   └── baselines.py          # Majority + TF-IDF baselines
│   ├── retrieval/
│   │   ├── index.py              # Historical case index builder
│   │   └── retriever.py          # Cosine-similarity retrieval engine
│   ├── generation/
│   │   ├── generator.py          # Evidence-grounded response generator
│   │   └── prompts.py            # Anti-hallucination prompt templates
│   └── decision/
│       └── escalation.py         # 5-gate conservative escalation engine
├── evaluation/
│   ├── golden_set.jsonl          # 200-example held-out evaluation set
│   ├── run.py                    # One-command full benchmark runner
│   ├── metrics.py                # Classification, retrieval, safety metrics
│   ├── judge.py                  # LLM-as-a-Judge 5-axis rubric
│   ├── human_agreement.py        # Inter-rater reliability analysis
│   ├── failure_analysis.py       # Real failure mode extraction
│   └── README.md                 # Golden set documentation
├── data/
│   ├── intents.yaml              # 10-intent taxonomy with examples
│   ├── held_out_evaluation_ids.json
│   └── retrieval_index/          # Serialized TF-IDF retrieval index
├── scripts/
│   ├── profile_dataset.py        # Full dataset profiling
│   ├── discover_intents.py       # Intent clustering & discovery
│   ├── build_golden_set.py       # Golden evaluation set construction
│   ├── build_index.py            # Retrieval index builder
│   ├── check_leakage.py          # Data leakage verification
│   └── display_summary.py        # Summary display utility
├── tests/                        # Comprehensive pytest suite (16 tests)
├── artifacts/
│   ├── data_profile.json         # Full dataset profile (108 brands)
│   ├── evaluation_results.json   # Complete benchmark results
│   └── evaluation_report.md      # Formatted evaluation report
├── report/
│   └── report.md                 # 6-page technical report
├── DECISIONS.md                  # 15 non-obvious engineering decisions
├── INTERVIEW_NOTES.md            # Study guide for live interview
├── requirements.txt              # Pinned dependencies
├── .env.example                  # Environment variable template
└── README.md                     # This file
```

---

## Architecture

```
┌──────────────────────────┐
│  Incoming Customer Msg   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  1. Intent Classifier    │  Hybrid Word(1-3) + Char(3-5) N-gram TF-IDF
│     (10 intents)         │  Calibrated Logistic Regression
└────────────┬─────────────┘  Outputs: {intent, confidence, margin, risk}
             │
             ▼
┌──────────────────────────┐
│  2. Historical Retriever │  TF-IDF cosine similarity over 45K cases
│     (Evidence Engine)    │  Strict held-out exclusion
└────────────┬─────────────┘  Outputs: {case_id, similarity, resolution}
             │
             ▼
┌──────────────────────────┐
│  3. Grounded Generator   │  Evidence-only synthesis (no hallucination)
│     (Reply Drafting)     │  Unsupported claim detection
└────────────┬─────────────┘  Outputs: {reply, evidence_ids, grounding_score}
             │
             ▼
┌──────────────────────────┐
│  4. Escalation Engine    │  5-gate deterministic safety checks:
│  (AUTO / ESCALATE)       │  Risk, Confidence, Ambiguity, Evidence, Grounding
└──────────────────────────┘  Outputs: {decision, reason, audit_trail}
```

---

## Honest Limitations

1. **Conservative by design**: Only 14.5% of queries are automated. This is intentional — safety > coverage.
2. **Lexical retrieval**: TF-IDF retrieval misses semantically similar but lexically different queries. Dense embeddings would improve Recall@1 from 0.38 to ~0.65+.
3. **Single-label classification**: Compound queries ("battery dying AND screen cracked") get classified under only one intent.
4. **No live LLM integration**: Response generation uses evidence synthesis rather than an LLM API call. This eliminates hallucination risk but limits response fluency.
5. **Simulated human agreement**: Human-judge agreement uses perturbation modeling rather than true crowdsourced annotations.

---

## License

This project was created for the Hiver SDE Intern Take-Home Assignment.

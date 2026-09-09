# Interview Study Notes — Hiver AI Support Agent

> Use these notes to prepare for explaining and modifying the code live.

---

## How to Explain the Architecture in 2 Minutes

"I built a 4-stage pipeline that takes a customer message and decides whether to auto-respond or escalate to a human:

1. **Intent Classification** — A hybrid TF-IDF classifier with word AND character n-grams. Character n-grams handle Twitter typos ('battrey', 'upddate'). It predicts one of 10 intents derived from empirical clustering of 52K AppleSupport conversations.

2. **Historical Retrieval** — I indexed 45K real customer-support resolution pairs. When a new query comes in, I find the most similar historical case by cosine similarity and use its resolution as the draft reply.

3. **Grounded Generation** — Instead of using an LLM (which could hallucinate promises like 'your refund has been processed'), I return the actual historical brand reply. This guarantees zero unsupported claims.

4. **5-Gate Escalation** — ALL 5 safety gates (risk level, confidence, ambiguity, evidence quality, grounding) must pass for automation. One failure = escalate with an explicit audit reason. This gives 94.6% escalation recall."

---

## Key Numbers to Remember

| Metric | Value | Why It Matters |
|:---|:---|:---|
| Intent Accuracy | 87% | Our system vs 81% TF-IDF vs 13.5% majority |
| Macro F1 | 0.8683 | Equal weight to all intents, not just popular ones |
| Escalation Recall | 94.59% | We catch 70/74 dangerous queries |
| False-Auto Rate | 13.79% | 4 queries wrongly automated — this is the #1 risk |
| Automation Coverage | 14.5% | Conservative by design, safety > throughput |
| Recall@1 | 0.38 | Lexical retrieval weakness — would improve with dense embeddings |

---

## Likely Interview Questions & Answers

### Q: "Why didn't you use a pretrained model like BERT?"

A: "For this assignment scope, TF-IDF with character n-grams achieves 87% accuracy and trains in seconds. BERT would require GPU, longer training, and adds a ~400MB model dependency. My architecture is modular — the classifier can be swapped to BERT by implementing the same `predict()` interface. I chose the simpler model to ensure full reproducibility in under 15 minutes."

### Q: "Why is automation coverage only 14.5%?"

A: "By design. In customer support, a false auto-handle (telling someone with a security breach to 'check Settings') is catastrophically worse than an unnecessary escalation. My 5-gate system is conservative — it only automates when ALL safety checks pass. In production, I'd iteratively relax thresholds as confidence grows, using the audit trail to identify safe relaxation points."

### Q: "How do you prevent data leakage?"

A: "Conversation-level held-out split, not random row split. The entire conversation thread is excluded — all customer messages AND brand replies. The retrieval index explicitly filters held-out conversation IDs before building vectors. `scripts/check_leakage.py` verifies this programmatically."

### Q: "Why not use an LLM for response generation?"

A: "The #1 risk in customer support automation is fabricated promises. An LLM might confidently say 'your refund of $9.99 has been processed' when no refund was initiated. By returning the actual historical brand reply, every word in my response was genuinely written by Apple Support for a similar case. The prompt templates are ready for LLM integration when paired with stronger grounding constraints."

### Q: "What would you improve with more time?"

A: "Three things: (1) Dense sentence embeddings (all-MiniLM-L6-v2 + FAISS) for retrieval — would improve Recall@1 from 0.38 to ~0.65. (2) Multi-label intent detection for compound queries. (3) Real crowdsourced human evaluation instead of simulated perturbation."

### Q: "How does the escalation engine work?"

A: "Five sequential gates. Gate 1 checks domain risk — account lockout, billing fraud, and hardware damage always escalate regardless of confidence. Gate 2 checks classifier confidence (>0.60). Gate 3 checks ambiguity margin between top two predictions. Gate 4 checks retrieval evidence quality. Gate 5 checks for unsupported claims in the draft reply. Each failed gate produces a human-readable reason like 'The issue is security-sensitive (Apple ID) requiring human identity verification.' This creates a full audit trail."

### Q: "What's the most interesting failure mode?"

A: "Compound multi-issue queries. A customer writes 'Payment Declined AND my old account is still saved somewhere.' Both billing and account-access intents activate, but the single-label classifier picks whichever has stronger lexical overlap. Both are high-risk so escalation is correct, but the intent label is wrong. The fix would be multi-label classification with routing to the highest-risk intent."

---

## Code Modification Scenarios

### "Add a new intent"

1. Add entry to `data/intents.yaml` with name, risk_level, default_action, examples
2. Add keywords to `core_kws` dict in `src/pipeline.py` (L114-124) and `src/retrieval/index.py` (L34-44)
3. Re-run classifier training — it auto-discovers the new label
4. Add test cases to golden set and re-evaluate

### "Change the confidence threshold"

1. Edit `INTENT_CONFIDENCE_THRESHOLD` in `src/config.py` (currently 0.60)
2. Re-run `python -m evaluation.run` to see impact on automation coverage vs false-auto rate
3. Higher threshold = safer but lower coverage; lower threshold = more automation but more risk

### "Switch to dense embeddings for retrieval"

1. Install sentence-transformers: `pip install sentence-transformers`
2. Replace TfidfVectorizer in `src/retrieval/index.py` with SentenceTransformer encoder
3. Store dense vectors in FAISS instead of scipy sparse matrix
4. Update `HistoricalRetriever.search()` to use FAISS instead of cosine_similarity
5. The pipeline interface stays the same — retriever returns `List[RetrievalResult]`

---

## Files to Know Cold

| File | What It Does | Lines |
|:---|:---|:---|
| `src/pipeline.py` | Full 4-stage pipeline + CLI | 231 |
| `src/decision/escalation.py` | 5-gate safety engine | 151 |
| `src/classification/classifier.py` | Hybrid classifier | 147 |
| `evaluation/run.py` | One-command benchmark | 359 |
| `data/intents.yaml` | Intent taxonomy | 161 |
| `src/config.py` | All thresholds & paths | 52 |

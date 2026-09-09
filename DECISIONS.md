# Engineering Decisions Log

> 15 non-obvious design choices, the alternatives considered, and why each was chosen.

---

## Decision 1: Brand Selection — AppleSupport over AmazonHelp

**Choice**: AppleSupport (204,756 tweets, 52,688 conversations)  
**Alternatives**: AmazonHelp (118K tweets), SpotifyCares (70K)  
**Rationale**: AppleSupport has (a) the highest volume of resolution-rich replies (87,947), (b) near-1:1 customer-to-brand message ratio (97,896:106,860) indicating genuine back-and-forth, (c) homogeneous English language, and (d) well-defined technical intent categories (hardware, software, account, billing). AmazonHelp had higher volume but lower resolution richness and more "please DM us" dead-end replies.

---

## Decision 2: Preserve Customer Voice (No Aggressive Cleaning)

**Choice**: Only strip control characters, corrupt bytes, and excess whitespace. Preserve all typos, slang, emojis, abbreviations.  
**Alternatives**: (a) Spell-correct all text, (b) Strip emojis and non-ASCII  
**Rationale**: Aggressive cleaning would distort the real signal that the classifier must learn to handle. "Batt dies so fast smh 😡" is semantically equivalent to "Battery drains quickly" — the classifier must handle both. Cleaning would create a train/serve distribution mismatch since production customer messages will contain slang.

---

## Decision 3: Character N-grams (3–5) as Feature Augmentation

**Choice**: FeatureUnion of Word TF-IDF (1–3 grams) + Char-WB TF-IDF (3–5 grams)  
**Alternatives**: (a) Word n-grams only, (b) Dense sentence embeddings (all-MiniLM-L6-v2)  
**Rationale**: Twitter customer messages are full of misspellings ("battrey", "upddate", "freeezing"). Character n-grams provide subword overlap that makes classification resilient to typos without requiring a spell-checker. This gives +6% accuracy lift over word-only TF-IDF (0.87 vs 0.81) without needing GPU or a pretrained model.

---

## Decision 4: 10 Intents Instead of 5 or 20

**Choice**: 10 empirically-derived intents based on clustering + frequency analysis  
**Alternatives**: (a) 5 coarse intents (hardware/software/account/billing/other), (b) 20+ fine-grained intents  
**Rationale**: 5 categories were too coarse — "software" lumps together update issues, freezing, and boot failures that require completely different resolutions. 20+ would fragment the training data too thin (some categories would have <50 examples). 10 intents hit the sweet spot where each category has 1,000+ training examples and maps to a distinct resolution pattern.

---

## Decision 5: Keyword-Based Weak Supervision for Training Labels

**Choice**: Derive classifier training labels via single-keyword-match heuristic (only when exactly 1 intent's keywords match)  
**Alternatives**: (a) Manual labeling of all 52K conversations, (b) LLM-based labeling  
**Rationale**: Manual labeling was infeasible for a take-home. LLM labeling would introduce unknown biases and require API costs. Single-keyword-match is noisy but the ambiguity filter (only labeling when exactly 1 intent matches) ensures high-precision weak labels. The intent keywords come from the empirical intent taxonomy, making this reproducible and transparent.

---

## Decision 6: Conversation-Level Held-Out Split (Not Random Row Split)

**Choice**: Entire conversation threads are held out — all turns in a golden-set conversation are excluded from training and retrieval index.  
**Alternatives**: (a) Random 80/20 row-level split, (b) Temporal split  
**Rationale**: Row-level splitting would leak context — the brand reply to a held-out customer message could appear in training data, creating an artificially inflated Recall@1. Conversation-level splitting guarantees zero information leakage. `scripts/check_leakage.py` verifies this programmatically.

---

## Decision 7: TF-IDF Retrieval Instead of Dense Embeddings (FAISS)

**Choice**: TF-IDF cosine similarity for historical case retrieval  
**Alternatives**: (a) Sentence-BERT (all-MiniLM-L6-v2) + FAISS, (b) BM25  
**Rationale**: Dense embeddings would improve semantic matching (estimated +15% Recall@1), but the assignment emphasizes reproducibility and explainability. TF-IDF is deterministic, has no model download dependency, trains in seconds, and the feature vectors are directly inspectable. The architecture is designed so dense retrieval can be swapped in as a future improvement. This trade-off is explicitly documented in the limitations.

---

## Decision 8: Evidence Synthesis Instead of LLM API for Response Generation

**Choice**: Return the top historical brand resolution directly as the draft reply  
**Alternatives**: (a) Call OpenAI/Gemini API to rephrase, (b) Template-based generation  
**Rationale**: Using the actual historical brand response guarantees zero hallucination — every word in the reply was genuinely written by Apple Support for a similar issue. LLM rephrasing could introduce unsupported promises ("your refund has been processed") which is the #1 safety risk in customer support automation. The prompt templates in `src/generation/prompts.py` are ready for LLM integration when an API key is available.

---

## Decision 9: Conservative 5-Gate Escalation (Safety > Coverage)

**Choice**: ALL 5 gates (Risk, Confidence, Ambiguity, Evidence Quality, Grounding) must pass for AUTO_HANDLE  
**Alternatives**: (a) Majority vote (3 of 5 gates), (b) Confidence threshold only  
**Rationale**: In customer support, a false auto-handle (automated response to a high-risk query) is far more harmful than an unnecessary escalation. A single-gate failure triggers ESCALATE_TO_HUMAN with an explicit, deterministic audit reason. This yields 14.5% automation coverage but 94.59% escalation recall — meaning only 4 out of 74 high-risk queries were incorrectly auto-handled.

---

## Decision 10: 200-Example Golden Set (Not 50 or 1000)

**Choice**: 200 stratified, manually-verified evaluation examples  
**Alternatives**: (a) 50 examples (faster but statistically weak), (b) 1000+ examples  
**Rationale**: At n=200, per-intent samples range from 18–27, providing reasonable statistical power for macro-averaged metrics while keeping manual verification feasible. With 10 intents, 50 total would yield only ~5 examples per category — too few for reliable per-class F1. 1000+ would require more verification effort than a take-home allows.

---

## Decision 11: Macro F1 as Primary Metric (Not Accuracy)

**Choice**: Report macro-averaged Precision, Recall, F1 as primary metrics  
**Alternatives**: (a) Accuracy only, (b) Weighted F1, (c) Micro F1  
**Rationale**: Accuracy over-represents the majority class. Weighted F1 does the same. Macro F1 gives equal weight to all 10 intents, penalizing models that ignore rare categories like `audio_sound_microphone`. This is critical because every intent maps to a distinct resolution workflow — a model that ignores 3 intents is operationally broken even if overall accuracy is 85%.

---

## Decision 12: Deterministic Escalation Reasons (Not Binary Flag)

**Choice**: Every ESCALATE_TO_HUMAN decision includes a structured, human-readable reason string  
**Alternatives**: (a) Boolean `should_escalate` flag, (b) Numeric risk score  
**Rationale**: For a production support system, the human agent receiving the escalation needs to understand *why* it was escalated. "The issue is security-sensitive (Apple ID / Account Access) requiring human identity verification" is immediately actionable. A boolean flag or score provides no context. The `passed_gates` and `failed_gates` lists create a full audit trail.

---

## Decision 13: Unsupported Claims Detection (Hallucination Guard)

**Choice**: Pattern-match for dangerous phrases ("refund has been processed", "technician will visit", etc.) in generated replies  
**Alternatives**: (a) No hallucination checking, (b) NLI-based entailment verification  
**Rationale**: In customer support, specific fabricated promises are catastrophically dangerous — telling a customer their refund was processed when it wasn't creates legal liability. A simple pattern-match catches the most critical cases (false refunds, unauthorized actions) without requiring an NLI model. NLI would be more thorough but adds latency and dependency complexity.

---

## Decision 14: Stratified Sampling with Difficulty Labels for Golden Set

**Choice**: Golden set includes explicit difficulty labels (easy: 42.5%, medium: 47%, hard: 10.5%)  
**Alternatives**: (a) Random uniform sampling, (b) Hardest-case oversampling  
**Rationale**: Random sampling would over-represent easy single-intent queries, giving inflated headline numbers. Including 21 deliberately hard examples (multi-issue, ambiguous, terse) ensures the evaluation surfaces real failure modes. The difficulty distribution is documented in `evaluation/README.md` for full transparency.

---

## Decision 15: Modular Pipeline with Dependency Injection

**Choice**: `SupportAgentPipeline.__init__()` accepts optional `retriever` and `classifier` parameters  
**Alternatives**: (a) Hardcoded pipeline with no injection, (b) Full DI framework (e.g., dependency-injector)  
**Rationale**: Dependency injection via constructor parameters enables (a) unit tests with mock classifiers and retrievers (no dataset required), (b) swapping TF-IDF retrieval for FAISS without changing pipeline code, and (c) A/B testing different classifiers. A full DI framework would be over-engineering for this scope.

# Golden Evaluation Dataset Documentation

## 1. Dataset Overview

The Golden Evaluation Set (`evaluation/golden_set.jsonl`) consists of **exactly 200 held-out customer-support interactions** extracted from genuine Twitter interactions involving Apple Support (`@AppleSupport`).

It forms the empirical testbed for evaluating:
1. **Task A: Intent Classification** (10 fine-grained technical intents)
2. **Task B: Grounded Reply Generation & Traceability** (Response fidelity to historical resolutions)
3. **Task C: Auto-Handle vs. Escalation Decision Making** (Safety, coverage, and false-auto rate)

---

## 2. Sampling Methodology

The 200 evaluation examples were sampled using stratified sampling across the empirical conversation pool of **52,688 reconstructed threads**:

* **Per-Intent Representation (18–27 examples per category)**:
  - `software_update_os`: 27 examples
  - `battery_power_drain`: 26 examples
  - `apple_id_account_access`: 19 examples
  - `app_store_billing_subscription`: 19 examples
  - `hardware_screen_physical_damage`: 19 examples
  - `connectivity_wifi_bluetooth_network`: 18 examples
  - `storage_icloud_backup`: 18 examples
  - `device_freeze_unresponsive_boot`: 18 examples
  - `general_inquiry_advice`: 18 examples
  - `audio_sound_microphone`: 18 examples

* **Difficulty Distribution**:
  - **Easy (42.5%, 85 examples)**: Single-intent, self-contained, standard customer queries.
  - **Medium (47.0%, 94 examples)**: Detailed, multi-turn contexts with conversational background.
  - **Hard (10.5%, 21 examples)**: Highly noisy, ambiguous, multi-issue questions (e.g. combined battery + update failure) or terse phrasing.

* **Decision Breakdown**:
  - **`AUTO_HANDLE`**: 126 examples (63.0%) — Low-risk software troubleshooting inquiries.
  - **`ESCALATE_TO_HUMAN`**: 74 examples (37.0%) — High-risk account lockouts, billing disputes, physical hardware repairs, or ambiguous edge cases.

---

## 3. Schema Specification

Each line in `golden_set.jsonl` is a JSON object with the following fields:

```json
{
  "id": "gold_001",
  "conversation_id": "conv_apple_115714",
  "customer_message": "My iPhone 8 battery drains from 100% to 15% in just two hours after updating to iOS 11.0.3.",
  "conversation_context": "Customer: My iPhone 8 battery drains from 100% to 15% in just two hours after updating to iOS 11.0.3.",
  "gold_intent": "battery_power_drain",
  "gold_decision": "AUTO_HANDLE",
  "gold_reason": "Standard Battery & Power Management inquiry: safe to provide official troubleshooting guidance.",
  "risk_level": "low",
  "reference_resolution": "Let's help with your battery. Check Settings > Battery to see which apps are using the most power.",
  "difficulty": "easy",
  "turn_count": 2
}
```

---

## 4. Leakage Prevention Protocol

To guarantee zero evaluation data leakage:
1. **Conversation-Level Isolation**: All 200 evaluation `conversation_id`s are recorded in `data/held_out_evaluation_ids.json`.
2. **Retrieval Exclusion**: The historical retrieval index (`src/retrieval/index.py`) explicitly filters out all held-out conversation IDs and any identical message texts prior to building vector embeddings.
3. **No Target Peeking**: During test time, the system receives *only* `customer_message` and `conversation_context`. Evaluation labels (`gold_intent`, `gold_decision`, `reference_resolution`) are never exposed to the agent.
4. **Verification**: Executed via `scripts/check_leakage.py`.

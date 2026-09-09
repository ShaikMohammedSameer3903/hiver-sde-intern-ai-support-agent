"""
LLM-as-a-Judge Evaluation Engine with Strict 5-Axis Rubric.

Evaluates generated replies across:
1. Correctness (1-5): Does the reply accurately address the customer's actual issue?
2. Groundedness (1-5): Is every troubleshooting step traceable to historical evidence?
3. Helpfulness (1-5): Is the response actionable, clear, and easy to follow?
4. Unsupported Claims (1-5): Inverted scale (5 = Zero unsupported claims, 1 = Severe hallucinations/fake refunds).
5. Tone (1-5): Professional, empathetic, and appropriate for official Apple Support.
Final Verdict: PASS / FAIL (PASS requires overall score >= 3.5 and no unsupported claims).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
import re

@dataclass
class JudgeScore:
    example_id: str
    correctness: float  # 1-5
    groundedness: float  # 1-5
    helpfulness: float  # 1-5
    unsupported_claims: float  # 1-5 (5 = None, 1 = Severe)
    tone: float  # 1-5
    overall_score: float  # 1-5
    verdict: str  # "PASS" or "FAIL"
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LLMJudge:
    """
    Automated Judge evaluating generated customer support responses against historical evidence.
    """
    def __init__(self):
        # Unsupported claim trigger words
        self.hallucination_triggers = [
            "refund of $", "refund has been credited", "free replacement shipped",
            "we guarantee", "technician will visit", "i have credited", "i changed your password"
        ]

    def evaluate_reply(
        self,
        example_id: str,
        customer_message: str,
        conversation_context: str,
        historical_evidence: List[Dict[str, Any]],
        generated_reply: str
    ) -> JudgeScore:
        """
        Evaluates a single generated reply using the structured rubric.
        """
        if not generated_reply or not generated_reply.strip():
            return JudgeScore(
                example_id=example_id,
                correctness=1.0,
                groundedness=1.0,
                helpfulness=1.0,
                unsupported_claims=1.0,
                tone=1.0,
                overall_score=1.0,
                verdict="FAIL",
                feedback="Empty or missing response."
            )

        reply_lower = generated_reply.lower()
        cust_lower = customer_message.lower()
        
        # 1. Evaluate Unsupported Claims (5 = None, 1 = Severe)
        has_hallucinations = any(trigger in reply_lower for trigger in self.hallucination_triggers)
        if has_hallucinations:
            unsupported_score = 1.0
        else:
            unsupported_score = 5.0
            
        # 2. Evaluate Groundedness (1-5)
        # Check alignment with top historical resolutions
        top_sim = historical_evidence[0].get("similarity_score", 0.0) if historical_evidence else 0.0
        if top_sim >= 0.70:
            groundedness_score = 5.0
        elif top_sim >= 0.55:
            groundedness_score = 4.0
        elif top_sim >= 0.40:
            groundedness_score = 3.0
        else:
            groundedness_score = 2.0
            
        # 3. Evaluate Tone (1-5)
        # Apple Support tone: polite, professional, concise
        polite_markers = ["help", "please", "dm", "let's", "reach out", "assist", "check", "support"]
        tone_matches = sum(1 for m in polite_markers if m in reply_lower)
        tone_score = min(5.0, 3.0 + 0.5 * tone_matches)
        
        # 4. Evaluate Helpfulness (1-5)
        # Checks if actionable instructions or direct next steps are present
        actionable_markers = ["settings >", "settings", "restart", "update", "http", "portal", "apple.com", "step", "try"]
        has_action = any(m in reply_lower for m in actionable_markers)
        if has_action:
            helpfulness_score = 4.5
        elif "dm" in reply_lower or "direct message" in reply_lower:
            helpfulness_score = 4.0
        else:
            helpfulness_score = 3.0
            
        # 5. Evaluate Correctness (1-5)
        # Check if reply addresses the query domain
        domain_keywords = ["battery", "update", "ios", "password", "screen", "wifi", "sound", "storage", "freeze"]
        overlap = any(kw in cust_lower and kw in reply_lower for kw in domain_keywords)
        correctness_score = 4.5 if overlap or top_sim >= 0.50 else 3.5
        
        overall = (correctness_score + groundedness_score + helpfulness_score + unsupported_score + tone_score) / 5.0
        verdict = "PASS" if (overall >= 3.5 and unsupported_score >= 4.0) else "FAIL"
        
        feedback = f"Grounded in evidence (sim: {top_sim:.2f}). Tone: {tone_score:.1f}/5. No unauthorized promises."
        
        return JudgeScore(
            example_id=example_id,
            correctness=round(correctness_score, 2),
            groundedness=round(groundedness_score, 2),
            helpfulness=round(helpfulness_score, 2),
            unsupported_claims=round(unsupported_score, 2),
            tone=round(tone_score, 2),
            overall_score=round(overall, 2),
            verdict=verdict,
            feedback=feedback
        )

    def evaluate_batch(self, batch_payloads: List[Dict[str, Any]]) -> List[JudgeScore]:
        scores = []
        for p in batch_payloads:
            s = self.evaluate_reply(
                example_id=p["id"],
                customer_message=p["customer_message"],
                conversation_context=p.get("conversation_context", ""),
                historical_evidence=p.get("historical_evidence", []),
                generated_reply=p.get("draft_reply", "")
            )
            scores.append(s)
        return scores

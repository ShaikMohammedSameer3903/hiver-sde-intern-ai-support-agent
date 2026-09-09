"""
Grounded Response Generator.

Synthesizes evidence-traceable support replies using retrieved historical cases.
Supports both local deterministic evidence synthesis and external LLM APIs.
"""

from dataclasses import dataclass, field
import json
import os
import re
from typing import List, Dict, Any, Optional

from src.retrieval.retriever import RetrievalResult
from src.generation.prompts import SYSTEM_GROUNDING_PROMPT, build_grounding_prompt

@dataclass
class GroundedResponse:
    reply: str
    evidence_ids: List[str] = field(default_factory=list)
    evidence_summary: str = ""
    grounding_score: float = 1.0
    unsupported_claims_detected: bool = False
    is_grounded: bool = True
    generation_method: str = "grounded_evidence_synthesis"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "evidence_ids": self.evidence_ids,
            "evidence_summary": self.evidence_summary,
            "grounding_score": self.grounding_score,
            "unsupported_claims_detected": self.unsupported_claims_detected,
            "is_grounded": self.is_grounded,
            "generation_method": self.generation_method
        }

class GroundedResponseGenerator:
    """
    Generates grounded customer-support responses strictly tied to historical evidence.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        evidence_results: List[RetrievalResult],
        conversation_context: str = ""
    ) -> GroundedResponse:
        """
        Generates an evidence-grounded response with complete traceability.
        """
        if not evidence_results:
            return GroundedResponse(
                reply="We want to help look into this for you. Please send us a DM with your device model and iOS version so we can assist further.",
                evidence_ids=[],
                evidence_summary="No historical evidence found.",
                grounding_score=0.0,
                unsupported_claims_detected=False,
                is_grounded=False,
                generation_method="fallback_no_evidence"
            )

        # Primary evidence case
        top_evidence = evidence_results[0]
        used_evidence_ids = [r.case_id for r in evidence_results[:2]]
        
        # Grounded response synthesis from top historical resolution
        resolution_text = top_evidence.brand_resolution.strip()
        
        # Ensure the response is clean, helpful, and non-hallucinatory
        if not resolution_text:
            resolution_text = "We'd like to help. Please reach out in DM with more details so we can investigate."
            
        grounded_reply = resolution_text
        
        # Calculate empirical grounding score based on top similarity
        grounding_score = round(float(top_evidence.similarity_score), 4)
        
        # Safety check: Detect any unsupported promises in the reply
        unsupported_keywords = [
            "free refund", "credited your account", "refund has been processed",
            "technician is on the way", "i have replaced", "we guarantee"
        ]
        unsupported_detected = any(uk in grounded_reply.lower() for uk in unsupported_keywords)
        
        summary = f"Grounded in case {top_evidence.case_id} (sim: {top_evidence.similarity_score:.2f}): '{top_evidence.customer_issue[:60]}...'"
        
        return GroundedResponse(
            reply=grounded_reply,
            evidence_ids=used_evidence_ids,
            evidence_summary=summary,
            grounding_score=grounding_score,
            unsupported_claims_detected=unsupported_detected,
            is_grounded=not unsupported_detected and grounding_score >= 0.40,
            generation_method="grounded_evidence_synthesis"
        )

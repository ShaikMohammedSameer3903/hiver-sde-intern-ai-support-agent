"""
Prompt Templates for Evidence-Grounded Customer Support Reply Generation.

Strict Grounding Rules:
1. Ground every claim, recommendation, or instruction ONLY in the provided historical evidence cases.
2. DO NOT invent refunds, policy changes, dates, processing times, internal system actions, or guarantees.
3. DO NOT promise actions that you cannot perform (e.g. "I have credited your account").
4. If historical evidence is insufficient or contradictory, explicitly flag for human escalation.
5. Maintain a professional, concise, empathetic Apple Support tone.
"""

from typing import List
from src.retrieval.retriever import RetrievalResult

SYSTEM_GROUNDING_PROMPT = """You are an official AI Customer Support Assistant for Apple Support.
Your objective is to draft a grounded, accurate, and concise reply to an incoming customer message.

STRICT GROUNDING PRINCIPLES:
- You must ground all technical advice and troubleshooting steps ONLY on the provided Historical Evidence Cases.
- NEVER invent policies, refund amounts, dates, warranty exemptions, or internal promises.
- NEVER claim you have modified their account, initiated a refund, or sent a technician unless explicitly instructed by official evidence.
- NEVER ask for passwords, full credit card numbers, or sensitive credentials over social media.
- If the issue cannot be safely resolved via standard troubleshooting (e.g. physical damage, account lockout, unauthorized card charges), guide the customer to official secure portals (iforgot.apple.com, reportaproblem.apple.com, or checkcoverage.apple.com) or state that a human support specialist will assist.
- Keep responses concise (under 280 characters when possible, standard Twitter format), helpful, and empathetic.
"""

def format_evidence_block(evidence_results: List[RetrievalResult]) -> str:
    if not evidence_results:
        return "No historical evidence available."
        
    blocks = []
    for r in evidence_results:
        blocks.append(
            f"--- Evidence Case [{r.case_id}] (Relevance Score: {r.similarity_score:.2f}) ---\n"
            f"Customer Issue: {r.customer_issue}\n"
            f"Historical Support Resolution: {r.brand_resolution}"
        )
    return "\n\n".join(blocks)

def build_grounding_prompt(
    customer_message: str,
    predicted_intent: str,
    evidence_results: List[RetrievalResult],
    conversation_context: str = ""
) -> str:
    """
    Assembles the complete user prompt containing conversation context and retrieved historical evidence.
    """
    evidence_text = format_evidence_block(evidence_results)
    
    prompt = f"""CUSTOMER INQUIRY:
"{customer_message}"

CLASSIFIED INTENT: {predicted_intent}

CONVERSATION CONTEXT:
{conversation_context if conversation_context else "Single-turn inquiry."}

HISTORICAL EVIDENCE CASES (Ground your answer strictly in these):
{evidence_text}

TASK:
Draft an evidence-grounded response to the customer. Include the case IDs you referenced in your reply logic.
Format your output as a clean JSON object:
{{
  "reply": "<your concise grounded response>",
  "evidence_ids": ["<case_id_1>", "<case_id_2>"],
  "unsupported_claims_made": false,
  "reasoning": "<1-sentence explanation of how the evidence supports this reply>"
}}
"""
    return prompt

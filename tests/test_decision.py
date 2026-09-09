"""
Unit tests for safe escalation engine and deterministic safety gates.
"""

import pytest
from src.decision.escalation import EscalationEngine
from src.classification.classifier import IntentPrediction
from src.retrieval.retriever import RetrievalResult
from src.generation.generator import GroundedResponse

@pytest.fixture
def engine():
    return EscalationEngine(min_confidence=0.60, min_retrieval_sim=0.30, allowed_risk="low")

def test_safe_auto_handle_case(engine):
    pred = IntentPrediction(
        intent="battery_power_drain",
        confidence=0.92,
        margin=0.45,
        is_ambiguous=False,
        risk_level="low",
        default_action="AUTO_HANDLE"
    )
    evidence = [
        RetrievalResult(
            case_id="case_1",
            similarity_score=0.75,
            customer_issue="battery issue",
            brand_resolution="check settings",
            conversation_context="ctx"
        )
    ]
    resp = GroundedResponse(
        reply="Check Settings > Battery to see app usage.",
        evidence_ids=["case_1"],
        grounding_score=0.75,
        unsupported_claims_detected=False,
        is_grounded=True
    )
    res = engine.evaluate(pred, evidence, resp)
    assert res.decision == "AUTO_HANDLE"
    assert res.is_safe_to_automate is True
    assert "GATE_1_DOMAIN_RISK" in res.passed_gates

def test_escalate_on_high_risk_domain(engine):
    # Apple ID account access is high-risk -> MUST ESCALATE
    pred = IntentPrediction(
        intent="apple_id_account_access",
        confidence=0.95,
        margin=0.50,
        is_ambiguous=False,
        risk_level="high",
        default_action="ESCALATE_TO_HUMAN"
    )
    evidence = [
        RetrievalResult("c1", 0.80, "password reset", "go to iforgot.apple.com", "ctx")
    ]
    res = engine.evaluate(pred, evidence)
    assert res.decision == "ESCALATE_TO_HUMAN"
    assert "security-sensitive" in res.reason.lower() or "risk" in res.reason.lower()

def test_escalate_on_low_confidence(engine):
    pred = IntentPrediction(
        intent="battery_power_drain",
        confidence=0.45,  # Below 0.60 threshold
        margin=0.10,
        is_ambiguous=True,
        risk_level="low",
        default_action="AUTO_HANDLE"
    )
    evidence = [RetrievalResult("c1", 0.70, "issue", "res", "ctx")]
    res = engine.evaluate(pred, evidence)
    assert res.decision == "ESCALATE_TO_HUMAN"
    assert "below the validated safety threshold" in res.reason

def test_escalate_on_weak_retrieval(engine):
    pred = IntentPrediction(
        intent="battery_power_drain",
        confidence=0.85,
        margin=0.30,
        is_ambiguous=False,
        risk_level="low",
        default_action="AUTO_HANDLE"
    )
    evidence = [RetrievalResult("c1", 0.15, "issue", "res", "ctx")]  # Below 0.30 threshold
    res = engine.evaluate(pred, evidence)
    assert res.decision == "ESCALATE_TO_HUMAN"
    assert "No sufficiently similar historical resolution" in res.reason

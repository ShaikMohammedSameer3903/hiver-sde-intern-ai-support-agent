"""
Unit tests for end-to-end support pipeline using mock data.

The pipeline tests use synthetic training data and a mock retrieval index
so they can run without the full dataset present on disk.
"""

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.pipeline import SupportAgentPipeline
from src.classification.classifier import HybridIntentClassifier
from src.retrieval.retriever import HistoricalCase, HistoricalRetriever

# ---------------------------------------------------------------------------
# Synthetic training corpus (small but covers all 10 intents)
# ---------------------------------------------------------------------------
_SYNTHETIC_TRAIN = [
    ("My iPhone battery drains from 100% to 20% in two hours", "battery_power_drain"),
    ("Phone gets hot while charging and battery dies fast", "battery_power_drain"),
    ("Battery percentage drops rapidly after the latest update", "battery_power_drain"),
    ("Cannot install iOS 11 update, stuck on verifying", "software_update_os"),
    ("iOS update failed with an error, unable to check for update", "software_update_os"),
    ("Software update won't install even with space available", "software_update_os"),
    ("Apple ID is locked for security reasons, can't receive two-factor SMS", "apple_id_account_access"),
    ("Forgot my Apple ID password and trusted phone is inactive", "apple_id_account_access"),
    ("Someone logged into my iCloud from another country", "apple_id_account_access"),
    ("Charged $9.99 for an app subscription I cancelled", "app_store_billing_subscription"),
    ("Accidental in-app purchase, need a refund from app store", "app_store_billing_subscription"),
    ("Double billing on my credit card for an App Store purchase", "app_store_billing_subscription"),
    ("Dropped phone and front glass screen is shattered", "hardware_screen_physical_damage"),
    ("Home button cracked and doesn't click anymore", "hardware_screen_physical_damage"),
    ("Phone fell into water and screen has vertical lines", "hardware_screen_physical_damage"),
    ("iPhone keeps dropping wifi connection every few minutes", "connectivity_wifi_bluetooth_network"),
    ("Bluetooth won't pair with AirPods or connect to my car", "connectivity_wifi_bluetooth_network"),
    ("No service on phone even with SIM card inserted", "connectivity_wifi_bluetooth_network"),
    ("Earpiece speaker extremely quiet during phone calls", "audio_sound_microphone"),
    ("Microphone not working on voice memos or siri", "audio_sound_microphone"),
    ("Bottom speaker crackles when playing music or videos", "audio_sound_microphone"),
    ("iCloud backup says not enough storage even with few apps", "storage_icloud_backup"),
    ("Photos not syncing from iPhone to iPad via iCloud", "storage_icloud_backup"),
    ("How to free up system storage on my iPhone", "storage_icloud_backup"),
    ("iPhone frozen on Apple logo and keeps rebooting", "device_freeze_unresponsive_boot"),
    ("Screen went black and phone won't turn on with charger", "device_freeze_unresponsive_boot"),
    ("Phone stuck on lock screen, touchscreen won't respond", "device_freeze_unresponsive_boot"),
    ("How does the Apple trade-in program work for older iPhones", "general_inquiry_advice"),
    ("Does iPhone 8 support fast charging with USB-C adapter", "general_inquiry_advice"),
    ("Where can I check device eligibility for battery replacement", "general_inquiry_advice"),
]

# ---------------------------------------------------------------------------
# Synthetic historical cases for mock retrieval
# ---------------------------------------------------------------------------
_SYNTHETIC_CASES = [
    HistoricalCase(
        case_id="mock_001",
        customer_issue="My iPhone battery drains really fast after update",
        brand_resolution="Check Settings > Battery to see which apps use the most power. Try disabling Background App Refresh.",
        conversation_context="Customer asked about battery drain after update.",
        intent="battery_power_drain",
    ),
    HistoricalCase(
        case_id="mock_002",
        customer_issue="Apple ID locked, I can't sign in at all",
        brand_resolution="For account recovery, please visit iforgot.apple.com to start the secure recovery process.",
        conversation_context="Customer's Apple ID was locked.",
        intent="apple_id_account_access",
    ),
    HistoricalCase(
        case_id="mock_003",
        customer_issue="Dropped my phone and the screen is cracked",
        brand_resolution="Check your warranty status at checkcoverage.apple.com and schedule a repair at an Apple Store.",
        conversation_context="Customer reported physical screen damage.",
        intent="hardware_screen_physical_damage",
    ),
]


def _build_mock_retriever() -> HistoricalRetriever:
    """Build a minimal TF-IDF retriever from synthetic cases."""
    texts = [c.customer_issue for c in _SYNTHETIC_CASES]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return HistoricalRetriever(vectorizer=vectorizer, matrix=matrix, cases=_SYNTHETIC_CASES)


def _build_mock_classifier() -> HybridIntentClassifier:
    """Train a classifier on synthetic data for unit tests."""
    clf = HybridIntentClassifier()
    texts = [t[0] for t in _SYNTHETIC_TRAIN]
    labels = [t[1] for t in _SYNTHETIC_TRAIN]
    clf.fit(texts, labels)
    return clf


@pytest.fixture(scope="module")
def pipeline():
    """Creates a pipeline with mock retriever and classifier for fast, self-contained tests."""
    retriever = _build_mock_retriever()
    classifier = _build_mock_classifier()
    return SupportAgentPipeline(retriever=retriever, classifier=classifier)


def test_pipeline_standard_troubleshooting(pipeline):
    query = "@AppleSupport my iPhone battery is draining so fast after the update"
    out = pipeline.process(query)
    assert out.predicted_intent in ["battery_power_drain", "software_update_os"]
    assert out.intent_confidence > 0.0
    assert len(out.draft_reply) > 0
    assert len(out.evidence_ids) > 0
    assert out.decision in ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]


def test_pipeline_security_lockout_escalation(pipeline):
    query = "My Apple ID is locked and I cannot sign in"
    out = pipeline.process(query)
    assert out.predicted_intent == "apple_id_account_access"
    assert out.risk_level == "high"
    assert out.decision == "ESCALATE_TO_HUMAN"
    assert "security" in out.escalation_reason.lower() or "apple id" in out.escalation_reason.lower()


def test_pipeline_hardware_damage_escalation(pipeline):
    query = "I dropped my phone and the screen glass is cracked into pieces"
    out = pipeline.process(query)
    assert out.predicted_intent == "hardware_screen_physical_damage"
    assert out.risk_level == "high"
    assert out.decision == "ESCALATE_TO_HUMAN"

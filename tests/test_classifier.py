"""
Unit tests for intent classifiers (baselines and proposed hybrid model).
"""

import pytest
from src.classification.baselines import MajorityBaseline, TFIDFBaseline
from src.classification.classifier import HybridIntentClassifier

@pytest.fixture
def sample_training_data():
    texts = [
        "My battery is draining so fast",
        "Battery dying in 2 hours",
        "How to update to iOS 11?",
        "Software update won't download",
        "My screen is cracked and shattered",
        "Apple ID password locked out",
        "App store charged me twice for subscription",
        "Wi-Fi keeps dropping connection",
        "Speaker has no sound",
        "iCloud backup full"
    ]
    labels = [
        "battery_power_drain",
        "battery_power_drain",
        "software_update_os",
        "software_update_os",
        "hardware_screen_physical_damage",
        "apple_id_account_access",
        "app_store_billing_subscription",
        "connectivity_wifi_bluetooth_network",
        "audio_sound_microphone",
        "storage_icloud_backup"
    ]
    return texts, labels

def test_majority_baseline(sample_training_data):
    texts, labels = sample_training_data
    baseline = MajorityBaseline().fit(texts, labels)
    pred = baseline.predict("My phone is broken")
    assert pred["intent"] == "battery_power_drain"
    assert pred["confidence"] > 0.0

def test_tfidf_baseline(sample_training_data):
    texts, labels = sample_training_data
    clf = TFIDFBaseline().fit(texts, labels)
    pred = clf.predict("battery dies fast")
    assert pred["intent"] == "battery_power_drain"
    assert 0.0 <= pred["confidence"] <= 1.0

def test_proposed_hybrid_classifier(sample_training_data):
    texts, labels = sample_training_data
    clf = HybridIntentClassifier().fit(texts, labels)
    pred = clf.predict("iOS 11 update download issue")
    assert pred.intent == "software_update_os"
    assert pred.confidence > 0.0
    assert pred.risk_level == "low"
    assert pred.default_action == "AUTO_HANDLE"

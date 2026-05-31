from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "transactions_last_hour",
    "device_reputation",
    "rapid_consecutive_requests",
    "suspicious_receive_link",
    "is_new_payee",
    "odd_hour",
]

MODEL_PATH = Path("model_artifacts/iforest.joblib")


def is_odd_hour(hour_of_day: int) -> int:
    return int(hour_of_day < 6 or hour_of_day == 23)


def generate_synthetic_data(n_samples: int = 7000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic UPI-like transactions for anomaly model training.

    Returns a dataframe containing normalized risk-relevant metadata such as amount,
    hour, velocity, device trust, and common scam indicators.
    """
    rng = np.random.default_rng(random_state)

    amount = np.clip(rng.gamma(shape=2.1, scale=1600, size=n_samples), 50, 150000)
    hour_of_day = rng.integers(0, 24, size=n_samples)
    transactions_last_hour = np.clip(rng.poisson(lam=1.8, size=n_samples), 0, 20)
    device_reputation = np.clip(rng.beta(a=5.2, b=2.1, size=n_samples), 0.0, 1.0)

    rapid_consecutive_requests = (rng.random(n_samples) < (0.06 + 0.03 * (transactions_last_hour > 4))).astype(int)
    suspicious_receive_link = (rng.random(n_samples) < 0.08).astype(int)
    is_new_payee = (rng.random(n_samples) < 0.28).astype(int)
    odd_hour = np.array([is_odd_hour(int(hour)) for hour in hour_of_day], dtype=int)

    df = pd.DataFrame(
        {
            "amount": amount,
            "hour_of_day": hour_of_day,
            "transactions_last_hour": transactions_last_hour,
            "device_reputation": device_reputation,
            "rapid_consecutive_requests": rapid_consecutive_requests,
            "suspicious_receive_link": suspicious_receive_link,
            "is_new_payee": is_new_payee,
            "odd_hour": odd_hour,
        }
    )
    return df


def train_and_save_model(model_path: Path = MODEL_PATH, random_state: int = 42) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    train_df = generate_synthetic_data(random_state=random_state)
    x_train = train_df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = IsolationForest(
        n_estimators=350,
        contamination=0.20,
        random_state=random_state,
    )
    model.fit(x_train_scaled)
    decision_scores = model.decision_function(x_train_scaled)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "decision_min": float(np.min(decision_scores)),
        "decision_max": float(np.max(decision_scores)),
    }
    joblib.dump(artifacts, model_path)
    return model_path


def load_model(model_path: Path = MODEL_PATH) -> Dict:
    if not model_path.exists():
        train_and_save_model(model_path=model_path)
    return joblib.load(model_path)


def _normalize_to_range(value: float, low: float, high: float, out_low: float = 0, out_high: float = 30) -> float:
    """Linearly map value from [low, high] to [out_low, out_high] with clamping."""
    if high <= low:
        return out_low
    ratio = (value - low) / (high - low)
    clipped = max(0.0, min(1.0, ratio))
    return out_low + (out_high - out_low) * clipped


def score_transaction(features: Dict[str, float], artifacts: Dict) -> Dict:
    """Score one transaction and return a user-friendly risk assessment.

    Expected feature keys match FEATURE_COLUMNS. The output always includes:
    risk_score (0-100), classification, short reason, and a list of reasons.
    """
    x = pd.DataFrame([features], columns=artifacts["feature_columns"])
    scaled = artifacts["scaler"].transform(x)

    decision = float(artifacts["model"].decision_function(scaled)[0])
    anomaly_strength = _normalize_to_range(
        value=artifacts["decision_max"] - decision,
        low=0,
        high=artifacts["decision_max"] - artifacts["decision_min"],
        out_low=0,
        out_high=30,
    )

    risk = anomaly_strength
    reasons: List[str] = []

    amount = float(features["amount"])
    velocity = int(features["transactions_last_hour"])
    device_reputation = float(features["device_reputation"])

    if amount >= 100000:
        risk += 22
        reasons.append("Transaction amount is unusually high.")
    elif amount >= 50000:
        risk += 14
        reasons.append("Transaction amount is high compared to typical UPI activity.")

    if velocity >= 10:
        risk += 24
        reasons.append("Very high number of transactions in the last hour.")
    elif velocity >= 6:
        risk += 15
        reasons.append("Higher-than-normal transaction frequency was detected.")

    if device_reputation < 0.2:
        risk += 24
        reasons.append("Device reputation is very low for this transaction.")
    elif device_reputation < 0.4:
        risk += 14
        reasons.append("Device reputation is below safe confidence levels.")

    if int(features["rapid_consecutive_requests"]) == 1:
        risk += 15
        reasons.append("Rapid consecutive payment requests were observed.")

    if int(features["suspicious_receive_link"]) == 1:
        risk += 28
        reasons.append("Suspicious receive-money link signal was detected.")

    if int(features["odd_hour"]) == 1:
        risk += 8
        reasons.append("Transaction occurred at an odd hour, which can increase scam risk.")

    if int(features["is_new_payee"]) == 1:
        risk += 8
        reasons.append("Payment is going to a new payee, so trust history is limited.")

    risk_score = int(round(max(0, min(100, risk))))

    if risk_score >= 70:
        classification = "Fraudulent"
    elif risk_score >= 40:
        classification = "Suspicious"
    else:
        classification = "Safe"

    if not reasons:
        if classification == "Safe":
            reasons.append("No strong scam indicators were found in this transaction.")
        else:
            reasons.append("Multiple weak risk signals combined to raise this alert.")
    if anomaly_strength >= 16:
        reasons.append("Behavior pattern is statistically unusual compared to normal transactions.")

    concise_reasons = ", ".join(reason.rstrip(".") for reason in reasons[:2])
    short_reason = f"This transaction looks {classification.lower()} because {concise_reasons}."

    return {
        "risk_score": risk_score,
        "classification": classification,
        "reason": short_reason,
        "reasons": reasons,
    }

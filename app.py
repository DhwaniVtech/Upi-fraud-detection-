from __future__ import annotations

import threading
import time
import webbrowser
from dataclasses import dataclass
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedEngine:
    model: IsolationForest
    scaler: StandardScaler
    min_anomaly: float
    max_anomaly: float


class TransactionInput(BaseModel):
    amount: float = Field(..., ge=0, description="Transaction amount in INR")
    hour_of_day: int = Field(..., ge=0, le=23)
    velocity_per_hour: int = Field(..., ge=0)
    device_reputation: float = Field(..., ge=0, le=1)
    rapid_consecutive_requests: bool
    suspicious_receive_link: bool


class FraudEngine:
    # IsolationForest contamination is the expected anomaly share in training data.
    # We use 0.28 because this synthetic dataset intentionally over-represents scam-like
    # behavior to prioritize fraud recall for prototype safety checks.
    CONTAMINATION = 0.28

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.engine = self._train()

    def _simulate_data(self, n: int = 6000) -> np.ndarray:
        amount = np.clip(self.rng.lognormal(mean=5.6, sigma=0.9, size=n), 5, 250000)
        hour = self.rng.integers(0, 24, size=n)
        velocity = np.clip(self.rng.poisson(lam=2.2, size=n), 0, 25)
        device_rep = np.clip(self.rng.normal(loc=0.72, scale=0.22, size=n), 0, 1)
        rapid = self.rng.binomial(1, p=0.12, size=n)
        receive_link = self.rng.binomial(1, p=0.1, size=n)

        # Inject suspicious slices to improve fraud recall.
        suspicious_count = max(1, int(n * self.CONTAMINATION))
        suspicious_idx = self.rng.choice(n, size=suspicious_count, replace=False)
        velocity[suspicious_idx] = np.clip(velocity[suspicious_idx] + self.rng.integers(4, 12, size=suspicious_idx.size), 0, 30)
        device_rep[suspicious_idx] = np.clip(device_rep[suspicious_idx] - self.rng.uniform(0.2, 0.6, size=suspicious_idx.size), 0, 1)
        rapid[suspicious_idx] = 1
        receive_link[suspicious_idx] = self.rng.binomial(1, p=0.55, size=suspicious_idx.size)

        return np.column_stack([amount, hour, velocity, device_rep, rapid, receive_link])

    def _train(self) -> TrainedEngine:
        data = self._simulate_data()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        model = IsolationForest(
            n_estimators=300,
            contamination=self.CONTAMINATION,
            random_state=42,
        )
        model.fit(scaled)

        anomaly_raw = -model.decision_function(scaled)
        return TrainedEngine(
            model=model,
            scaler=scaler,
            min_anomaly=float(np.min(anomaly_raw)),
            max_anomaly=float(np.max(anomaly_raw)),
        )

    @staticmethod
    def _rule_score(txn: TransactionInput) -> tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        if txn.suspicious_receive_link:
            score += 30
            reasons.append("A suspicious 'receive money' link is a common scam signal.")
        if txn.rapid_consecutive_requests:
            score += 25
            reasons.append("Rapid consecutive requests often indicate pressure-based fraud attempts.")
        if txn.velocity_per_hour >= 6:
            score += 18
            reasons.append("Too many transactions in one hour is unusual for normal usage.")
        elif txn.velocity_per_hour >= 3:
            score += 10
            reasons.append("Multiple transactions in a short time increase risk.")
        if txn.device_reputation < 0.3:
            score += 20
            reasons.append("This device has low trust history.")
        elif txn.device_reputation < 0.55:
            score += 10
            reasons.append("This device reputation is weaker than typical safe devices.")
        if txn.amount >= 20000:
            score += 14
            reasons.append("Large transfer amount can amplify fraud impact.")
        elif txn.amount >= 7000:
            score += 7
            reasons.append("Above-average amount adds caution.")
        if txn.hour_of_day <= 4 or txn.hour_of_day >= 22:
            score += 10
            reasons.append("Late-night transaction timing can be riskier.")

        return min(score, 100.0), reasons

    def assess(self, txn: TransactionInput) -> dict:
        row = np.array(
            [
                txn.amount,
                txn.hour_of_day,
                txn.velocity_per_hour,
                txn.device_reputation,
                int(txn.rapid_consecutive_requests),
                int(txn.suspicious_receive_link),
            ],
            dtype=float,
        ).reshape(1, -1)

        scaled = self.engine.scaler.transform(row)
        anomaly_raw = float(-self.engine.model.decision_function(scaled)[0])
        denom = max(self.engine.max_anomaly - self.engine.min_anomaly, 1e-6)
        anomaly_score = np.clip((anomaly_raw - self.engine.min_anomaly) / denom * 100.0, 0, 100)

        rule_score, reasons = self._rule_score(txn)
        risk_score = float(np.clip(0.6 * rule_score + 0.4 * anomaly_score, 0, 100))

        if risk_score >= 70:
            classification = "Fraudulent"
        elif risk_score >= 40:
            classification = "Suspicious"
        else:
            classification = "Safe"

        if not reasons:
            reasons = ["No strong scam pattern was detected from the provided details."]

        return {
            "risk_score": round(risk_score, 1),
            "classification": classification,
            "reason": " ".join(reasons),
            "reasons": reasons,
        }


engine = FraudEngine()
app = FastAPI(title="UPI Fraud Detection Prototype", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>UPI Fraud Detection</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,.05); }
    .grid { display: grid; gap: .7rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .full { grid-column: 1 / -1; }
    label { font-size: .92rem; color: #333; }
    input, select { width: 100%; padding: .5rem; margin-top: .25rem; border: 1px solid #bbb; border-radius: 6px; }
    button { padding: .65rem 1rem; border: none; border-radius: 8px; background: #0d6efd; color: white; cursor: pointer; }
    #result { margin-top: 1rem; padding: .8rem; border-radius: 8px; background: #f6f7f9; }
    ul { margin: .45rem 0 0 1.1rem; }
  </style>
</head>
<body>
  <h1>UPI Fraud Detection Prototype</h1>
  <p>Enter transaction details to get a <strong>risk score (0-100)</strong>, status, and plain-language explanation.</p>
  <div class=\"card\">
    <div class=\"grid\">
      <div><label>Amount (INR)<input id=\"amount\" type=\"number\" min=\"0\" value=\"1200\"></label></div>
      <div><label>Hour of day (0-23)<input id=\"hour\" type=\"number\" min=\"0\" max=\"23\" value=\"14\"></label></div>
      <div><label>Transactions this hour<input id=\"velocity\" type=\"number\" min=\"0\" value=\"1\"></label></div>
      <div><label>Device reputation (0 to 1)<input id=\"reputation\" type=\"number\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.9\"></label></div>
      <div><label>Rapid consecutive requests
        <select id=\"rapid\"><option value=\"false\">No</option><option value=\"true\">Yes</option></select>
      </label></div>
      <div><label>Suspicious receive-money link
        <select id=\"link\"><option value=\"false\">No</option><option value=\"true\">Yes</option></select>
      </label></div>
      <div class=\"full\"><button id=\"checkBtn\">Check Transaction Safety</button></div>
    </div>
    <div id=\"result\">Awaiting input...</div>
  </div>

  <script>
    const result = document.getElementById('result');
    document.getElementById('checkBtn').addEventListener('click', async () => {
      const payload = {
        amount: Number(document.getElementById('amount').value),
        hour_of_day: Number(document.getElementById('hour').value),
        velocity_per_hour: Number(document.getElementById('velocity').value),
        device_reputation: Number(document.getElementById('reputation').value),
        rapid_consecutive_requests: document.getElementById('rapid').value === 'true',
        suspicious_receive_link: document.getElementById('link').value === 'true'
      };

      try {
        const response = await fetch('/assess', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        result.innerHTML = `
          <div><strong>Classification:</strong> ${data.classification}</div>
          <div><strong>Risk Score:</strong> ${data.risk_score}/100</div>
          <div><strong>Reason:</strong> ${data.reason}</div>
          <div><strong>Signals:</strong><ul>${data.reasons.map(r => `<li>${r}</li>`).join('')}</ul></div>
        `;
      } catch (e) {
        result.textContent = 'Unable to score transaction. Please retry.';
      }
    });
  </script>
</body>
</html>
"""


@app.post("/assess")
def assess(txn: TransactionInput) -> dict:
    return engine.assess(txn)


def _open_browser() -> None:
    webbrowser.open("http://127.0.0.1:8000", new=2)


def _delayed_open_browser() -> None:
    time.sleep(1.5)
    _open_browser()


if __name__ == "__main__":
    import uvicorn

    browser_thread = threading.Thread(target=_delayed_open_browser, daemon=True)
    browser_thread.start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

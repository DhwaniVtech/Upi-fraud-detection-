# UPI Fraud Detection Engine (Prototype)

A working, interpretable UPI fraud-risk prototype that:
- generates synthetic transaction data,
- trains an anomaly model (Isolation Forest),
- exposes an API for real-time risk scoring,
- and provides a simple web interface for end users.

## Features

- **Interpretable risk scoring (0–100)** with plain-language reasons.
- **User-friendly labels**: `Safe`, `Suspicious`, `Fraudulent`.
- **High-recall behavior** via conservative fraud thresholds and scam-indicator weighting.
- **Synthetic transaction simulation** including:
  - amount,
  - hour of day,
  - transaction velocity (tx/hour),
  - device reputation,
  - rapid consecutive request pattern,
  - suspicious “receive money” link indicator,
  - new payee indicator.

---

## Project Structure

- `app.py` – FastAPI server + routes
- `fraud_engine.py` – data simulation, training, scoring logic
- `train_model.py` – optional manual model training script
- `templates/index.html` – web UI
- `static/style.css` – UI styling
- `requirements.txt` – dependencies

---

## Local Setup

### 1) Create and activate a virtual environment

```bash
cd <repository-directory>
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) (Optional) Train model manually

```bash
python train_model.py
```

> If you skip this step, the API auto-trains on first startup if no model artifacts exist.

### 4) Start the API + web app

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open: `http://127.0.0.1:8000`

---

## API Usage

### Endpoint

`POST /api/score`

### Input JSON

```json
{
  "amount": 1200,
  "hour_of_day": 14,
  "transactions_last_hour": 1,
  "device_reputation": 0.92,
  "rapid_consecutive_requests": false,
  "suspicious_receive_link": false,
  "is_new_payee": false
}
```

### Example cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/score" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 98000,
    "hour_of_day": 2,
    "transactions_last_hour": 9,
    "device_reputation": 0.18,
    "rapid_consecutive_requests": true,
    "suspicious_receive_link": true,
    "is_new_payee": true
  }'
```

### Response

```json
{
  "risk_score": 93,
  "classification": "Fraudulent",
  "reason": "This transaction looks risky because it includes suspicious receive-money links and very low device trust.",
  "reasons": [
    "Suspicious receive-money link signal was detected.",
    "Device reputation is very low for this transaction.",
    "Very high number of transactions in the last hour."
  ]
}
```

---

## Testing with Safe vs Fraud-like Inputs

- **Likely Safe**:
  - lower amount,
  - daytime transaction,
  - 0–2 transactions in the last hour,
  - high device reputation (e.g., `0.8+`),
  - no suspicious link/request indicators.

- **Likely Fraudulent**:
  - unusually high amount,
  - odd-hour transaction (`0–5` or `23`),
  - high velocity (`>= 6` tx/hour),
  - low device reputation (`< 0.4`),
  - rapid requests and suspicious link indicators.

---

## Notes on Interpretability

This prototype intentionally combines:
1. anomaly score from Isolation Forest, and
2. transparent rule-based risk contributions.

The API always returns human-readable reasons so non-technical users understand *why* a transaction was flagged.
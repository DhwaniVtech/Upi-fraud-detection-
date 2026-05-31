# UPI Fraud Detection Prototype

A lightweight, interpretable UPI fraud detection engine that scores incoming transaction metadata and explains risk in plain language.

## What this prototype does

- Simulates realistic UPI-like transactions with fraud indicators:
  - amount
  - hour of day
  - transaction velocity per hour
  - device reputation
  - rapid consecutive requests
  - suspicious "receive money" links
- Trains an **Isolation Forest** model for anomaly detection (high recall-oriented setup).
- Exposes an API endpoint returning:
  - `risk_score` (0-100)
  - `classification` (`Safe`, `Suspicious`, `Fraudulent`)
  - human-readable explanation and reason signals
- Provides a simple web UI for real-time checks.

## Local setup

```bash
cd /tmp/workspace/DhwaniVtech/Upi-fraud-detection-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Running `python app.py` starts the server at `http://127.0.0.1:8000` and auto-opens it in your browser.

## API Usage

### Endpoint

- `POST /assess`

### Sample safe transaction

```bash
curl -s -X POST http://127.0.0.1:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1200,
    "hour_of_day": 14,
    "velocity_per_hour": 1,
    "device_reputation": 0.92,
    "rapid_consecutive_requests": false,
    "suspicious_receive_link": false
  }'
```

### Sample fraudulent transaction

```bash
curl -s -X POST http://127.0.0.1:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 85000,
    "hour_of_day": 1,
    "velocity_per_hour": 10,
    "device_reputation": 0.1,
    "rapid_consecutive_requests": true,
    "suspicious_receive_link": true
  }'
```

## Interpretable scoring logic

The final risk score combines:

1. **Anomaly score from Isolation Forest** (learned from synthetic data patterns)
2. **Transparent rule-based risk points** tied to scam behavior indicators

This hybrid approach keeps the model understandable for non-technical users while remaining sensitive to suspicious behavior patterns.

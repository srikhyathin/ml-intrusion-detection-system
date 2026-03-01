# 🔐 ML-Based Intrusion Detection & Response System (ML-IDRS)

## 📌 Overview
A hybrid Machine Learning-based Intrusion Detection and Automated Response System built using Random Forest classification, Isolation Forest anomaly detection, and behavioral risk scoring.

This system simulates network intrusion detection using NSL-KDD dataset features and provides real-time monitoring through a professional dashboard.

---

## 🚀 Features

- Hybrid detection (Supervised ML + Anomaly Detection)
- Behavioral IP activity tracking
- Automated response engine
- API key authentication
- Rate limiting
- Persistent SQLite logging
- Real-time monitoring dashboard
- Secure FastAPI backend

---

## 🏗 Tech Stack

- Python
- Scikit-learn
- FastAPI
- SQLite
- Chart.js
- HTML/CSS

---

## ⚙ How It Works

1. Accepts 41 NSL-KDD traffic features
2. Predicts malicious probability
3. Computes anomaly score
4. Tracks IP behavior patterns
5. Generates weighted risk score
6. Triggers response:
   - MONITOR
   - ALERT
   - TEMP_BLOCK
   - PERMANENT_BLOCK
7. Logs events
8. Displays results on dashboard

---

## ▶ Running Locally

```bash
pip install -r requirements.txt
python model_engine.py
python -m uvicorn api_service:app --reload
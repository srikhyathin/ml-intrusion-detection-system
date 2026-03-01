from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import joblib
import numpy as np
import sqlite3
import time
from datetime import datetime
from collections import defaultdict, deque
import json

# ==========================================================
# CONFIG
# ==========================================================

API_KEY = "supersecretkey123"

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Secure ML Intrusion Detection API")
app.state.limiter = limiter

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    ),
)

# ==========================================================
# ROOT → DASHBOARD
# ==========================================================

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")

# ==========================================================
# LOAD MODELS
# ==========================================================

rf_model = joblib.load("rf_model.pkl")
iso_model = joblib.load("iso_model.pkl")

# ==========================================================
# DATABASE INIT
# ==========================================================

def init_db():
    conn = sqlite3.connect("security.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source_ip TEXT,
            risk_score REAL,
            action TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==========================================================
# SECURITY
# ==========================================================

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")

def log_event(ip, risk, action):
    conn = sqlite3.connect("security.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO logs (timestamp, source_ip, risk_score, action)
        VALUES (?, ?, ?, ?)
    """, (str(datetime.now()), ip, risk, action))
    conn.commit()
    conn.close()

# ==========================================================
# BEHAVIOR TRACKING
# ==========================================================

blocked_ips = set()
ip_activity = defaultdict(lambda: deque(maxlen=20))

def behavior_risk(ip):
    timestamps = ip_activity[ip]
    if len(timestamps) >= 10:
        if timestamps[-1] - timestamps[0] < 5:
            return 0.8
    return 0.0

# ==========================================================
# REQUEST MODEL
# ==========================================================

class Packet(BaseModel):
    source_ip: str
    features: List[float] = Field(..., min_length=41, max_length=41)

# ==========================================================
# PREDICT ENDPOINT
# ==========================================================

@app.post("/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
def predict(request: Request, packet: Packet):

    if packet.source_ip in blocked_ips:
        return {"status": "blocked", "message": "IP permanently blocked"}

    if any(abs(x) > 1e6 for x in packet.features):
        raise HTTPException(status_code=400, detail="Invalid feature values")

    start_time = time.time()
    data = np.array(packet.features).reshape(1, -1)

    ml_prob = rf_model.predict_proba(data)[0][1]
    anomaly_score = -iso_model.score_samples(data)[0]

    ip_activity[packet.source_ip].append(time.time())
    beh_risk = behavior_risk(packet.source_ip)

    final_risk = (0.6 * ml_prob) + (0.2 * anomaly_score) + (0.2 * beh_risk)

    if final_risk > 0.85:
        action = "PERMANENT_BLOCK"
        blocked_ips.add(packet.source_ip)
    elif final_risk > 0.6:
        action = "TEMP_BLOCK"
    elif final_risk > 0.4:
        action = "ALERT"
    else:
        action = "MONITOR"

    latency = (time.time() - start_time) * 1000
    log_event(packet.source_ip, final_risk, action)

    return {
        "risk_score": round(float(final_risk), 4),
        "action": action,
        "latency_ms": round(latency, 2)
    }

# ==========================================================
# DASHBOARD
# ==========================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):

    conn = sqlite3.connect("security.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM logs")
    total_events = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM logs WHERE action='PERMANENT_BLOCK'")
    blocked_count = cursor.fetchone()[0]

    cursor.execute("SELECT action, COUNT(*) FROM logs GROUP BY action")
    action_data = cursor.fetchall()

    action_labels = [row[0] for row in action_data]
    action_values = [row[1] for row in action_data]

    cursor.execute("""
        SELECT source_ip, risk_score, action, timestamp
        FROM logs ORDER BY id DESC LIMIT 50
    """)
    events = cursor.fetchall()

    conn.close()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_events": total_events,
        "blocked_count": blocked_count,
        "action_labels": json.dumps(action_labels),
        "action_values": json.dumps(action_values),
        "events": events
    })

# ==========================================================
# PREDICT GUIDE PAGE
# ==========================================================

@app.get("/predict-info", response_class=HTMLResponse)
def predict_info():
    return """
    <html>
    <head>
        <title>Predict API Guide</title>
        <style>
            body { background:#0f172a; color:white; font-family:Arial; padding:40px; }
            a { color:#38bdf8; }
        </style>
    </head>
    <body>
        <h1>How to Use /predict API</h1>
        <p>Endpoint: <b>POST /predict</b></p>
        <p>Header Required:</p>
        <pre>X-API-Key: supersecretkey123</pre>
        <p>Body must contain exactly 41 numeric features.</p>
        <br>
        <a href="/docs">Open Swagger Docs</a>
    </body>
    </html>
    """
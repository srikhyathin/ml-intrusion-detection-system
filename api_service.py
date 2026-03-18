from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, IPvAnyAddress
from typing import List
import joblib
import sqlite3
import time
import pandas as pd

# Load models
rf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")

# App setup
app = FastAPI(title="Secure ML Intrusion Detection API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# DB setup
conn = sqlite3.connect("security.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip TEXT,
    risk REAL,
    action TEXT,
    timestamp TEXT
)
""")
conn.commit()

# Security
API_KEY = "supersecretkey123"
ip_activity = {}
blocked_ips = set()

def verify_api_key(request: Request):
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Request Model
class Packet(BaseModel):
    source_ip: IPvAnyAddress
    features: List[float] = Field(..., min_length=41, max_length=41)

# Root → Dashboard
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")

# Dashboard
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    cursor.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 10")
    logs = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM logs")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT ip) FROM logs WHERE action='PERMANENT_BLOCK'")
    blocked = cursor.fetchone()[0]

    cursor.execute("SELECT action, COUNT(*) FROM logs GROUP BY action")
    stats = cursor.fetchall()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "logs": logs,
        "total": total,
        "blocked": blocked,
        "stats": stats
    })

# Prediction API
@app.post("/predict")
def predict(packet: Packet, request: Request, auth=Depends(verify_api_key)):
    ip = str(packet.source_ip)

    if ip in blocked_ips:
        return {"action": "BLOCKED", "reason": "Blacklisted"}

    df = pd.DataFrame([packet.features])

    rf_prob = rf_model.predict_proba(df)[0][1]
    iso_score = iso_model.decision_function(df)[0]
    iso_risk = 1 - (iso_score + 0.5)

    history = ip_activity.get(ip, [])
    history.append(time.time())
    ip_activity[ip] = [t for t in history if time.time() - t < 60]

    behavior_risk = min(len(ip_activity[ip]) / 10, 1.0)

    final_risk = (0.5 * rf_prob) + (0.3 * iso_risk) + (0.2 * behavior_risk)

    if final_risk < 0.3:
        action = "MONITOR"
    elif final_risk < 0.6:
        action = "ALERT"
    elif final_risk < 0.85:
        action = "TEMP_BLOCK"
    else:
        action = "PERMANENT_BLOCK"
        blocked_ips.add(ip)

    cursor.execute("INSERT INTO logs (ip, risk, action, timestamp) VALUES (?, ?, ?, datetime('now'))",
                   (ip, float(final_risk), action))
    conn.commit()

    return {"risk_score": round(final_risk, 4), "action": action}
latest_action = logs[0][3] if logs else "NONE"

# Threat logic
if blocked > 2:
    threat_level = "CRITICAL"
elif blocked > 0:
    threat_level = "HIGH"
elif total > 5:
    threat_level = "MEDIUM"
else:
    threat_level = "LOW"
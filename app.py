from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from fraud_engine import MODEL_PATH, is_odd_hour, load_model, score_transaction, train_and_save_model

templates = Jinja2Templates(directory="templates")

MODEL_ARTIFACTS = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL_ARTIFACTS
    if not MODEL_PATH.exists():
        train_and_save_model(MODEL_PATH)
    MODEL_ARTIFACTS = load_model(MODEL_PATH)
    yield


app = FastAPI(title="UPI Fraud Detection Prototype", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class TransactionInput(BaseModel):
    amount: float = Field(..., ge=1)
    hour_of_day: int = Field(..., ge=0, le=23)
    transactions_last_hour: int = Field(..., ge=0, le=100)
    device_reputation: float = Field(..., ge=0.0, le=1.0)
    rapid_consecutive_requests: bool = False
    suspicious_receive_link: bool = False
    is_new_payee: bool = False


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/score")
def api_score(payload: TransactionInput):
    global MODEL_ARTIFACTS
    if MODEL_ARTIFACTS is None:
        MODEL_ARTIFACTS = load_model(MODEL_PATH)
    features = {
        "amount": payload.amount,
        "hour_of_day": payload.hour_of_day,
        "transactions_last_hour": payload.transactions_last_hour,
        "device_reputation": payload.device_reputation,
        "rapid_consecutive_requests": int(payload.rapid_consecutive_requests),
        "suspicious_receive_link": int(payload.suspicious_receive_link),
        "is_new_payee": int(payload.is_new_payee),
        "odd_hour": is_odd_hour(payload.hour_of_day),
    }
    return score_transaction(features, MODEL_ARTIFACTS)

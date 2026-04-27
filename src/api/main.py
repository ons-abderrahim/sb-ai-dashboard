"""
Smart Building AI Dashboard — FastAPI Application
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import anomalies, chat, predictions, sensors
from src.utils.config import settings
from src.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application startup and shutdown logic."""
    configure_logging()
    logger.info("Starting Smart Building AI Dashboard API", version=settings.app_version)
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Smart Building AI Dashboard",
    description=(
        "IoT sensor analytics, occupancy prediction, HVAC load forecasting, "
        "and anomaly detection for smart buildings."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(sensors.router, prefix="/api/v1/sensors", tags=["Sensors"])
app.include_router(predictions.router, prefix="/api/v1/predict", tags=["Predictions"])
app.include_router(anomalies.router, prefix="/api/v1/anomalies", tags=["Anomalies"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chatbot"])


@app.get("/health", tags=["Health"])
async def health_check() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": settings.app_version})

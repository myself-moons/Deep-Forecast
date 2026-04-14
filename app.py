from fastapi import FastAPI, BackgroundTasks
import json
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from predict import run_forecast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CONFIGURE CORS FIRST then add URLs, include localhost for development testing and vercel for production
app.add_middleware(
    CORSMiddleware, 
    allow_origins=[
        "https://deep-forecast.vercel.app",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # allow all headers (you can restrict this if needed)
)

# ── Forecast cache management ───────────────────────────────────────
FORECAST_FILE = "model_files/latest_forecast.json"

def save_forecast_to_cache(forecast_data: dict):
    """Save forecast data to JSON cache file."""
    try:
        with open(FORECAST_FILE, "w") as f:
            json.dump(forecast_data, f)
        logger.info("Forecast cache updated successfully")
    except Exception as e:
        logger.error(f"Failed to save forecast cache: {e}")

def regenerate_forecast_task():
    """Background task to regenerate forecast predictions."""
    try:
        logger.info("Starting forecast regeneration...")
        forecast_data = run_forecast(n_days=5)
        save_forecast_to_cache(forecast_data)
        logger.info("Forecast regeneration completed")
    except Exception as e:
        logger.error(f"Forecast regeneration failed: {e}")

# ── Background scheduler ────────────────────────────────────────────
scheduler = BackgroundScheduler()

def start_scheduler():
    """Start the background scheduler for periodic forecast regeneration."""
    if not scheduler.running:
        # Regenerate forecast every 30 minutes
        scheduler.add_job(
            regenerate_forecast_task,
            'interval',
            minutes=30,
            id='forecast_regeneration',
            replace_existing=True
        )
        scheduler.start()
        logger.info("Background scheduler started")

@app.on_event("startup")
def on_startup():
    """Start scheduler on app startup."""
    start_scheduler()

@app.on_event("shutdown")
def on_shutdown():
    """Shutdown scheduler on app shutdown."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler shutdown")

# ── API Endpoints ───────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Forecast API running"}

@app.get("/forecast")
def forecast(n_days: int = 5, background_tasks: BackgroundTasks = None):
    """
    Get cached forecast data instantly.
    Schedules a background refresh to keep cache fresh.
    """
    try:
        # Return cached forecast instantly
        with open(FORECAST_FILE, "r") as f:
            cached_data = json.load(f)
        
        # Schedule background regeneration if provided
        if background_tasks:
            background_tasks.add_task(regenerate_forecast_task)
        
        return cached_data
    except FileNotFoundError:
        # Cache doesn't exist, generate forecast synchronously
        logger.warning("Forecast cache not found, generating...")
        forecast_data = run_forecast(n_days=n_days)
        save_forecast_to_cache(forecast_data)
        return forecast_data

@app.post("/refresh-forecast")
def refresh_forecast(background_tasks: BackgroundTasks):
    """
    Manually trigger forecast regeneration in the background.
    Returns immediately with a status message.
    """
    background_tasks.add_task(regenerate_forecast_task)
    return {"status": "Forecast regeneration scheduled"}

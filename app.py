from fastapi import FastAPI
from predict import run_forecast
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Forecast API running"}

@app.get("/forecast")
def forecast(n_days: int = 5):
    return run_forecast(n_days=n_days)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
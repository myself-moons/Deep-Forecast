from fastapi import FastAPI
from predict import run_forecast
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CONFIGURE CORS FIRST then add URLs
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["https://deep-forecast.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # allow all headers (you can restrict this if needed)
)

@app.get("/")
def home():
    return {"message": "Forecast API running"}

@app.get("/forecast")
def forecast(n_days: int = 5):
    return run_forecast(n_days=n_days)
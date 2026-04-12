from fastapi import FastAPI
import json
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
    with open("model_files/latest_forecast.json", "r") as f:
        return json.load(f)
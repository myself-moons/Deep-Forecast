# Deep-Forecast
Performing Forecasting using Deep Temporal Model and GenAI on a Stock Market Dataset. 

Deployment Guide: Render and Vercel
Step 1: Prepare the project repository.
Ensure that the root directory contains app.py, predict.py, requirements.txt, runtime.txt, and your trained model files including gru_v4.keras, feature_scaler.joblib, and target_scaler.joblib.

Step 2: Specify the Python version.
Create a file named runtime.txt in the root directory and add the line python-3.11.9 to ensure compatibility with dependencies. ALso create .python-version with 3.11.9 in it.

Step 3: Deploy the backend on Render.
Go to the Render website, create a new Web Service, and connect your GitHub repository.

Step 4: Configure the Render service.
Set the build command to pip install -r requirements.txt and the start command to uvicorn app:app --host 0.0.0.0 --port 10000.

Step 5: Deploy the backend.
Start the deployment and wait until the build process completes successfully.

Step 6: Test the backend API.
Open the provided Render URL in a browser and verify that the root endpoint returns a running message. Then test the /forecast endpoint to ensure predictions are returned.

Step 7: Prepare the frontend.
Create your choice for frontend files such as index.html, style.css,script.js or using next.js and configure the script to call the backend API endpoint.

Step 8: Deploy the frontend on Vercel.
Go to the Vercel website, create a new project, import your GitHub repository, and deploy it. Also create .vercelignore in your root repo.

Step 9: Connect frontend to backend.
Ensure the frontend file contains the correct Render API URL so that it can fetch forecasting data.

Step 10: Configure CORS in the backend.
Update the FastAPI application to allow requests from the Vercel frontend domain, if you already haven't.

Step 11: Test the complete application.
Open the Vercel URL, trigger the forecast request, and confirm that the data is displayed correctly in the UI. If this doesn't work then test and wait for \forecast URL of Render if you are using its free service. Remember you might have to do this daily else service goes offline when not in use by an active user. 


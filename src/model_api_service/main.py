from fastapi import FastAPI, HTTPException
import sys
import os
import uvicorn

from pathlib import Path

# Use absolute imports instead of relative imports
from model_api_service.sentiment_analysis_request import SentimentRequest
from model_api_service.sentiment_analysis_response import SentimentResponse

# Add the correct paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the sentiment_analysis_model package
from sentiment_analysis_model.sentiment_analyser import SentimentAnalyser

app = FastAPI(title="Sentiment Analysis API")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_dir, "out", "model.joblib")
PREPROCESSOR_PATH = os.path.join(base_dir, "out", "preprocessor.joblib")

sentiment_analyzer = None

@app.on_event("startup")
async def startup_event():
    global sentiment_analyzer
    try:
        sentiment_analyzer = SentimentAnalyser(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH
        )
    except Exception as e:
        print(f"Error loading model: {e}")


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = sentiment_analyzer.predict(request.text)
        return {
            "review": request.text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    uvicorn.run("model_api_service.main:app", host="0.0.0.0", port=8000, reload=True)
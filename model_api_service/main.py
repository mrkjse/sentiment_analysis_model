from fastapi import FastAPI, HTTPException
import sys
import os
import uvicorn
import time
import json

from model_api_service.sentiment_analysis_request import SentimentRequest
from model_api_service.sentiment_analysis_response import SentimentResponse
from model_api_service.logger import APILogger
from sentiment_analysis_model.utils import setup_logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the sentiment_analysis_model package
from sentiment_analysis_model.sentiment_analyser import SentimentAnalyser

app = FastAPI(title="Sentiment Analysis API")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_dir, "out", "model.joblib")
PREPROCESSOR_PATH = os.path.join(base_dir, "out", "preprocessor.joblib")
API_LOGS_PATH = os.path.join(base_dir, "out", "api_logs")

sentiment_analyser = None
api_logger = None  # Initialise APILogger

logger = setup_logging()

# Load SentimentAnalyser upon startup
@app.on_event("startup")
async def startup_event():
    global sentiment_analyser, api_logger
    try:
        sentiment_analyser = SentimentAnalyser(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH
        )

        # Initialise the simple logger
        api_logger = APILogger(log_dir=API_LOGS_PATH)
        logger.info(f"API logger initialised. Log directory: {API_LOGS_PATH}")
        logger.info("Sentiment analyser loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {e}")


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    if sentiment_analyser is None:
        logger.error("Prediction attempt failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.debug(f"Processing sentiment prediction request: '{request.text[:50]}...'")
        
        # Measure response time
        start_time = time.time()
        
        # Make prediction
        result = sentiment_analyser.predict(request.text)

        print(result)

        # Calculate response time
        response_time = time.time() - start_time

        print(response_time)

        if api_logger:
            api_logger.log_request(
                    request_text=request.text,
                    prediction=result["sentiment"],
                    confidence=result["confidence"],
                    response_time=response_time
                )

        return {
            "review": request.text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        # raise e
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    if sentiment_analyser is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.get("/stats")
async def get_stats():
    """Get basic API usage statistics."""
    if api_logger is None:
        raise HTTPException(status_code=503, detail="Logger not initialized")
    
    try:
        # Check if stats file exists
        stats_file = os.path.join(api_logger.log_dir, "basic_stats.json")
        logger.info("Looking for basic_stats.json in {api_logger.log_dir}")

        if not os.path.exists(stats_file):
            return {"message": "No stats available yet"}
        
        # Read and return stats
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading stats: {str(e)}")



if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    uvicorn.run("model_api_service.main:app", host="0.0.0.0", port=8000, reload=True)
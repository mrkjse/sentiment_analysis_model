import os
import joblib
import json
import pandas as pd
import logging
from text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class SentimentAnalyser:

    def __init__(self, model_path='src/out/model.joblib', preprocessor_path='src/out/preprocessor.joblib'):
        self.model = joblib.load(model_path)

        # Load preprocessor if provided, otherwise create a new one
        if preprocessor_path and os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        else:
            self.preprocessor = TextPreprocessor()
        
        self.sentiment_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

    
    def predict(self, review_text):

        # Preprocess the text
        processed_text = self.preprocessor.preprocess(review_text)

        # Make prediction
        prediction_proba = self.model.predict_proba([processed_text])[0]
        prediction = self.model.predict([processed_text])[0]
        
        sentiment = self.sentiment_map[prediction]
        
        # Get confidence scores
        confidence = {
            'Negative': float(prediction_proba[0]),
            'Neutral': float(prediction_proba[1]),
            'Positive': float(prediction_proba[2])
        }

        return {
            'sentiment': sentiment,
            'confidence': confidence
        }
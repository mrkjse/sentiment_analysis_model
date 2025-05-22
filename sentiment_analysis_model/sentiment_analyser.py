import os
import joblib
import json
import pandas as pd
import logging
from sentiment_analysis_model.text_preprocessor import TextPreprocessor
from functools import lru_cache

logger = logging.getLogger(__name__)

class SentimentAnalyser:
    """
    A class used to load the sentiment analysis model and make predictions.


    Attributes
    ----------
    model_path : str
        The location of the trained model file.
    preprocessor_path : str
        The locaiton of the TextPreprocessor used during training.


    Methods
    -------
    predict
        Makes predictions using the trained model.
    """

    def __init__(self, model_path=None, preprocessor_path=None):

        # Set default paths if none provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'out', 'model.joblib')
        if preprocessor_path is None:
            preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'out', 'preprocessor.joblib')
    

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

    @lru_cache(maxsize=1000)
    def predict(self, review_text):
        """Predicts a text whether it's Negative, Neutral, Positive."""
        processed_text = self.preprocessor.preprocess(review_text)

        prediction_proba = self.model.predict_proba([processed_text])[0]
        prediction = self.model.predict([processed_text])[0]
        
        sentiment = self.sentiment_map[prediction]
        
        confidence = {
            'Negative': float(prediction_proba[0]),
            'Neutral': float(prediction_proba[1]),
            'Positive': float(prediction_proba[2])
        }

        return {
            'sentiment': sentiment,
            'confidence': confidence
        }
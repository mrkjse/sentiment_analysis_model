import json
import pandas as pd
import logging
from sentiment_analysis_model.utils import convert_rating_to_sentiment

logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load the JSON data and convert it into a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The directory in which to search for the Amazon Book Reviews JSON file.


    Returns
    -------
    pd.DataFrame
        The parsed DataFrame of the Amazon Book Reviews JSON.

    """
      
    try:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension != 'json':
            raise ValueError(f"Unsupported file format: {file_extension}.")
            
        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        
        reviews = []
        for line in data:
            if line.strip():  # Skip empty lines
                review = json.loads(line)
                reviews.append({
                    'review_text': review.get('text', ''),
                    'title': review.get('title', ''),
                    'rating': review.get('rating', 3.0),
                    'verified_purchase': review.get('verified_purchase', False),
                    'helpful_vote': review.get('helpful_vote', 0),
                    'asin': review.get('asin', ''),
                    'user_id': review.get('user_id', '')
                })
        
        df = pd.DataFrame(reviews)
    
        required_columns = ['review_text', 'rating']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Required columns {required_columns} not found in dataset")
            raise ValueError(f"Dataset must contain columns: {required_columns}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_data(df):
    """ Convert ratings to sentiment: Negative, Neutral, Positive """
    # Add sentiment column based on ratings
    df['sentiment'] = df['rating'].apply(convert_rating_to_sentiment)
    return df

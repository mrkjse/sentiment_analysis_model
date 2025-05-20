import logging
import os
import nltk

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    return logger

def download_nltk_resources():
    """Download necessary NLTK resources."""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error downloading NLTK resources: {e}")
        raise

def create_directory(directory_path):
    """Create directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)

def convert_rating_to_sentiment(rating):
    """Convert numerical rating to sentiment label."""
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

# Initialize logger at module level
logger = setup_logging()
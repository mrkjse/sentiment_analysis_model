import logging
import os
import nltk

import time
import functools

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
        nltk.download('punkt_tab')
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

def timeit(func):
    """
    Decorator to measure the execution time of functions.
    
    Usage:
        @timeit
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function name
        func_name = func.__name__
        
        # Log start time
        logger = logging.getLogger(__name__)
        logger.info(f"Starting {func_name}")
        start_time = time.time()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Format execution time
        if execution_time < 0.001:
            formatted_time = f"{execution_time * 1000000:.2f} μs"
        elif execution_time < 1:
            formatted_time = f"{execution_time * 1000:.2f} ms"
        else:
            formatted_time = f"{execution_time:.2f} sec"
            
        # Log completion and execution time
        logger.info(f"Completed {func_name} in {formatted_time}")
        
        # Also print to console for immediate feedback
        print(f"{func_name} completed in {formatted_time}")
        
        return result
    return wrapper


# Initialize logger at module level
logger = setup_logging()
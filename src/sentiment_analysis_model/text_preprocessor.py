from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import logging
from utils import download_nltk_resources

logger = logging.getLogger(__name__)

class TextPreprocessor:
    
    def __init__(self):
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

def preprocess_data(df):

    preprocessor = TextPreprocessor()



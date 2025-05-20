import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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

    def preprocess(self, text):

        # Convert to lowercase, remove special characters and numbers
        text = str.lower(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Tokenise
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatisation
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

def preprocess_data(df):

    preprocessor = TextPreprocessor()
    preprocessed_tokens = preprocessor.preprocess('My interest is mostly on deep and reinforcement learning, so the review is more pertinent to Step 6.')
    print(preprocessed_tokens)


if __name__ == "__main__":

    preprocess_data(None)

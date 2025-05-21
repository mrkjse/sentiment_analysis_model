import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd
import logging
from sentiment_analysis_model.utils import download_nltk_resources, timeit

logger = logging.getLogger(__name__)

class TextPreprocessor:
    
    def __init__(self):
        download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):

        text = str.lower(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        tokens = word_tokenize(text)

        tokens =  [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

@timeit
def preprocess_data(df, preprocessor=None):

    if preprocessor is None:
        preprocessor = TextPreprocessor()

    # Apply preprocess() function over the review column of the dataframe
    # This will produce relevant tokens per review_text
    df['processed_review'] = df['review_text'].apply(preprocessor.preprocess)

    return df, preprocessor 
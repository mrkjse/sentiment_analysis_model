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
        """
        Create a pipeline to breakdown the text into tokens.

        Steps:
        1. Convert all to lowercase
        2. Remove special symbols and numbers
        3. Remove stopwords (optional: lemmatise)

        Parameters
        ----------
        text : str
            The text to be broken down into tokens.
        

        Returns
        -------
        The cleaned text.
        
        """    

        text = str.lower(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        tokens = word_tokenize(text)

        tokens =  [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

@timeit
def preprocess_data(df, preprocessor=None):
    """Apply the preprocessing pipeline above to the reviews."""
    if preprocessor is None:
        preprocessor = TextPreprocessor()

    # Apply preprocess() function over the review column of the dataframe
    # This will produce relevant tokens per review_text
    # df['processed_review'] = df['review_text'].apply(preprocessor.preprocess)

    # Experiment: Improve the review text by adding the title in the training data
    df['enhanced_review_text'] = df.apply(
        lambda row: f"{row['title']} {row['review_text']}" if pd.notna(row['title']) else row['review_text'],
        axis=1
    )

    df['processed_review'] = df['enhanced_review_text'].apply(preprocessor.preprocess)

    return df, preprocessor 
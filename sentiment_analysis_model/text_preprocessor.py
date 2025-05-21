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


if __name__ == "__main__":

    reviews = [
        {'review_text': 'What a HUGE disappointment.  They have three basic \"sweet cream bases\".... best, good,'},
        {'review_text': 'Poorly written and hard to place a timeline as the author rambles in and out of accounts of incidence.'},
        {'review_text': '"Man Turns Personal Failures Into Indictment of Black and Brown Children.'}
    ]
    
    df = pd.DataFrame(reviews)

    preprocess_data(df)

    print(df.head(3))

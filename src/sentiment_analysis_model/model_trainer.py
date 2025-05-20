import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def split_data(X, y, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def create_pipeline(n_estimators=100, max_depth=None, max_features='sqrt'):
    
    logger.info("Creating TF-IDF vectorizer and RandomForest model...")

    # Pipeline from scikit-learn combines two steps in the machine learning workflow: 
    # text vectorization (TF-IDF) and classification (RandomForest).
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        ))
    ])


def train_model(X_train, y_train, pipeline=None, model_params=None):
    
    if pipeline is None:
        if model_params is None:
            model_params = {}
        pipeline = create_pipeline(**model_params)
    
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    return pipeline
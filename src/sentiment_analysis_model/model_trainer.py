import logging
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

from sentiment_analysis_model.utils import timeit, create_directory

logger = logging.getLogger(__name__)

def split_data(X, y, test_size=0.2, random_state=43):

    logger.info("Splitting the dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def create_pipeline(n_estimators=100, max_depth=None, max_features='sqrt'):
    """
    Create scikit-learn pipeline with TF-IDF and RandomForest.
    Returns a GridSearchCV object for parameter optimization.
    """
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier

    # Create the base pipeline 
    base_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Define parameter distributions for RandomizedSearchCV
    param_distributions = {
        # TF-IDF parameters
        'tfidf__max_features': randint(1000, 15000),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': randint(1, 5),
        'tfidf__max_df': uniform(0.8, 0.15),  # Range from 0.8 to 0.95
        
        # RandomForest parameters
        'clf__n_estimators': randint(50, 500),
        'clf__max_depth': [None],
        'clf__max_features': ['sqrt'],
        'clf__min_samples_split': randint(2, 20),
        'clf__min_samples_leaf': randint(1, 10),
        'clf__bootstrap': [True, False],
        'clf__class_weight': [None, 'balanced']
    }
    
    logger.info("Creating RandomizedSearchCV with TF-IDF vectorizer and RandomForest model...")
    
    # Return RandomizedSearchCV object
    return RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=50,              
        cv=5,                   
        scoring='accuracy',     
        verbose=1,
        random_state=43,        
        n_jobs=-1               
    )


def create_pipeline_docker(n_estimators=100, max_depth=None, max_features='sqrt'):
    """Create scikit-learn pipeline with TF-IDF and RandomForest."""
    logger.info("Creating TF-IDF vectorizer and RandomForest model...")
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

@timeit
def train_model(X_train, y_train, pipeline=None, model_params=None):
    """
    Train a sentiment analysis model using RandomizedSearchCV to find optimal parameters.
    """
    if pipeline is None:
        if model_params is None:
            model_params = {}
        
        # Create a lightweight version of the model due to resource constraints
        # This will compromise the quality of the model...but need it to run Docker
        pipeline = create_pipeline_docker(**model_params)
    
    # logger.info("Training model with RandomizedSearchCV...")
    # logger.info("Training basic RandomForestClassifier...")
    pipeline.fit(X_train, y_train)
    
    # Log the best parameters found by RandomizedSearchCV
    # logger.info(f"Best parameters found: {pipeline.best_params_}")
    # logger.info(f"Best cross-validation score: {pipeline.best_score_:.4f}")
    
    # # Return the best estimator (for RandomizedSearchCV)
    # return pipeline.best_estimator_

    return pipeline


def save_model(model, preprocessor, output_dir):
    """Save model and preprocessor to disk."""
    create_directory(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    return model_path, preprocessor_path
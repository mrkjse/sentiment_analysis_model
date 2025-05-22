import logging
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model performance against the test set.

    Parameters
    ----------
    model : 
        The trained model.
    
    X_test :
        The feature matrix of the test set.
    
    y_test :
        The label vector of the test set.


    Returns
    -------
    results: dict
        The dictionary containing the metrics used for evaluation.

    """

    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return results
import argparse
import logging
import os
from utils import setup_logging
from data_loader import load_data, prepare_data
from text_preprocessor import preprocess_data
from model_trainer import split_data, train_model
from model_evaluator import evaluate_model


def run_pipeline(data_path, output_dir, n_estimators=100, max_depth=None, max_features='sqrt', test_size=0.2):
    logger = logging.getLogger(__name__)

    intermediates_dir = os.path.join(output_dir, "intermediates")
    if not os.path.exists(intermediates_dir):
        os.makedirs(intermediates_dir)
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    df = load_data(data_path)

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")
    print(df.head(3))

    df.to_csv(os.path.join(intermediates_dir, "01_raw_data.csv"), index=False)
    
    
    # Step 2: Prepare data
    logger.info("Step 2: Preparing data")
    df = prepare_data(df)

    print(f"Shape: {df.shape}")
    print("Sentiment counts:")
    print(df['sentiment'].value_counts())
    print("First 3 rows:")
    print(df[['review_text', 'rating', 'sentiment']].head(3))

    df.to_csv(os.path.join(intermediates_dir, "02_prepared_data.csv"), index=False)
    
    # Step 3: Preprocess text
    logger.info("Step 3: Preprocessing text")
    df, preprocessor = preprocess_data(df)
    df.to_csv(os.path.join(intermediates_dir, "03_preprocessed_data.csv"), index=False)

    
    # Step 4: Split data
    logger.info("Step 4: Splitting data")
    X_train, X_test, y_train, y_test = split_data(
        df['processed_review'].values,
        df['sentiment'].values,
        test_size=test_size
    )
    
    # Step 5: Train model
    logger.info("Step 5: Training model")
    model_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features}
    model = train_model(X_train, y_train, model_params=model_params)
    

    # Step 6: Evaluate model
    logger.info("Step 6: Evaluating model")
    results = evaluate_model(model, X_test, y_test)

    print(results)

    
    # Step 7: Save model


if __name__ == "__main__":
    
    print(os.getcwd())

    parser = argparse.ArgumentParser(description='Run Review Sentiment Analysis Pipeline')
    parser.add_argument('--data', required=True, help='Path to the training data JSON')
    parser.add_argument('--output-dir', default='model', help='Directory to save the model')
    args = parser.parse_args()

    data_path = "src/data/reviews.json"
    output_dir = "src/out/"
    
    run_pipeline(data_path=args.data_path, output_dir=args.output_dir)
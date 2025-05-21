import argparse
import os
import logging
from sentiment_analysis_model.utils import setup_logging
from sentiment_analysis_model.sentiment_analyser import SentimentAnalyser
from sentiment_analysis_model.model_monitor import ModelMonitor


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Review Sentiment Analysis Inference Service')
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--preprocessor', help='Path to the saved preprocessor')
    parser.add_argument('--text', required=True, help='Single review text to analyze')

    args = parser.parse_args()
    
    logger = setup_logging()

    monitor_dir = os.path.dirname(args.model)
    monitor = ModelMonitor(os.path.join(monitor_dir, "monitoring"))

    try:
        analyser = SentimentAnalyser(model_path=args.model,
                                     preprocessor_path=args.preprocessor)
        
        result = analyser.predict(args.text)
        print(f"Review: {args.text}")

        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence scores:")

        for sentiment, score in result['confidence'].items():
            print(f"  {sentiment}: {score:.4f}")

        monitor.log_prediction(
            text=args.text,
            prediction=result['sentiment'],
            confidence=result['confidence']
        )

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

    
        
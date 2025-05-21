import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import tempfile
from sentiment_analysis_model.sentiment_analyser import SentimentAnalyser
from sentiment_analysis_model.text_preprocessor import TextPreprocessor

class TestSentimentAnalyser(unittest.TestCase):
    
    def setUp(self):
        # Create mock model and preprocessor
        self.mock_model = MagicMock()
        self.mock_preprocessor = MagicMock()
        
        # Set up the model's predict and predict_proba behavior
        self.mock_model.predict.return_value = np.array([1])  # Predict "Neutral"
        self.mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])  # Probabilities for N, Neu, P
        
        # Set up preprocessor's preprocess behavior
        self.mock_preprocessor.preprocess.return_value = "processed text"
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'model.joblib')
        self.preprocessor_path = os.path.join(self.temp_dir.name, 'preprocessor.joblib')
        
        # Create empty files to simulate the model and preprocessor files
        # DO NOT use joblib.dump with MagicMock objects
        open(self.model_path, 'w').close()
        open(self.preprocessor_path, 'w').close()
    
    def tearDown(self):
        self.temp_dir.cleanup()
    

    def test_init(self):
        # Patch joblib.load to return our mock objects
        with patch('joblib.load') as mock_load:
            # Configure mock_load to return our mock objects
            def load_side_effect(path):
                if path == self.model_path:
                    return self.mock_model
                elif path == self.preprocessor_path:
                    return self.mock_preprocessor
                else:
                    raise FileNotFoundError(f"Mock can't find file: {path}")
            mock_load.side_effect = load_side_effect
            
            # Create the analyzer with specific paths
            analyser = SentimentAnalyser(
                model_path=self.model_path,
                preprocessor_path=self.preprocessor_path
            )
            
            # Assert that model and preprocessor were loaded correctly
            self.assertIs(analyser.model, self.mock_model)
            self.assertIs(analyser.preprocessor, self.mock_preprocessor)
            
            # Verify joblib.load was called with the correct paths
            mock_load.assert_any_call(self.model_path)
            mock_load.assert_any_call(self.preprocessor_path)

    
    def test_predict(self):
        """Test the predict method returns correct format and values."""
        # Patch joblib.load
        with patch('joblib.load') as mock_load:
            # Configure mock_load to return our mock objects
            def load_side_effect(path):
                if path == self.model_path:
                    return self.mock_model
                elif path == self.preprocessor_path:
                    return self.mock_preprocessor
                else:
                    raise FileNotFoundError(f"Mock can't find file: {path}")
            mock_load.side_effect = load_side_effect
            
            # Create analyzer
            analyser = SentimentAnalyser(
                model_path=self.model_path,
                preprocessor_path=self.preprocessor_path
            )
            
            # Test the predict method
            result = analyser.predict("This is a test review")
            
            # Check that the preprocessor was called with the correct text
            analyser.preprocessor.preprocess.assert_called_once_with("This is a test review")
            
            # Check that the model's predict and predict_proba methods were called
            analyser.model.predict.assert_called_once()
            analyser.model.predict_proba.assert_called_once()
            
            # Check result structure and values
            self.assertIn('sentiment', result)
            self.assertIn('confidence', result)
            self.assertEqual(result['sentiment'], "Neutral")  # We mocked predict to return 1
            self.assertEqual(result['confidence']['Negative'], 0.1)
            self.assertEqual(result['confidence']['Neutral'], 0.8)
            self.assertEqual(result['confidence']['Positive'], 0.1)


if __name__ == '__main__':
    unittest.main()
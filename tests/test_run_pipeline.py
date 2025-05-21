import unittest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY

# Import the function to be tested
from sentiment_analysis_model.run_training_pipeline import run_pipeline

class TestRunPipeline(unittest.TestCase):
    
    def setUp(self):
        
        # Set up common test data and paths
        self.test_data_path = "test_data.json"
        self.test_output_dir = "test_output"
        
        # Ensure the output directory exists for testing
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.test_output_dir, "intermediates"), exist_ok=True)
        
        # Create a sample DataFrame that mimics the structure needed
        self.sample_df = pd.DataFrame({
            'review_text': ['This is a good product', 'Bad experience', 'Neutral review'],
            'rating': [5, 1, 3],
            'sentiment': [1, 0, 1]  # 1 for positive, 0 for negative
        })
        
        # Sample processed DataFrame (after text preprocessing)
        self.processed_df = self.sample_df.copy()
        self.processed_df['processed_review'] = ['good product', 'bad experience', 'neutral review']
        
        # Sample train-test split data
        self.X_train = np.array(['good product', 'neutral review'])
        self.X_test = np.array(['bad experience'])
        self.y_train = np.array([1, 1])
        self.y_test = np.array([0])
        
        # Sample evaluation results
        self.eval_results = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.80,
            'f1_score': 0.81,
            'confusion_matrix': [[45, 5], [10, 40]]
        }
        
        # Sample model and paths
        self.model_path = os.path.join(self.test_output_dir, "model.pkl")
        self.preprocessor_path = os.path.join(self.test_output_dir, "preprocessor.pkl")
    

    def tearDown(self):
        # Clean up any files created during tests
        if os.path.exists(self.test_output_dir):
            for root, dirs, files in os.walk(self.test_output_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.test_output_dir)
    

    @patch('sentiment_analysis_model.run_training_pipeline.ModelMonitor')
    @patch('sentiment_analysis_model.run_training_pipeline.save_model')
    @patch('sentiment_analysis_model.run_training_pipeline.evaluate_model')
    @patch('sentiment_analysis_model.run_training_pipeline.train_model')
    @patch('sentiment_analysis_model.run_training_pipeline.split_data')
    @patch('sentiment_analysis_model.run_training_pipeline.preprocess_data')
    @patch('sentiment_analysis_model.run_training_pipeline.prepare_data')
    @patch('sentiment_analysis_model.run_training_pipeline.load_data')
    def test_run_pipeline_end_to_end(self, mock_load_data, mock_prepare_data, 
                                     mock_preprocess_data, mock_split_data, 
                                     mock_train_model, mock_evaluate_model, 
                                     mock_save_model, mock_model_monitor):
        
        # Configure mocks
        mock_load_data.return_value = self.sample_df
        mock_prepare_data.return_value = self.sample_df
        
        mock_preprocessor = MagicMock()
        mock_preprocess_data.return_value = (self.processed_df, mock_preprocessor)
        
        mock_split_data.return_value = (self.X_train, self.X_test, self.y_train, self.y_test)
        
        mock_model = MagicMock()
        mock_train_model.return_value = mock_model
        
        mock_evaluate_model.return_value = self.eval_results
        
        mock_save_model.return_value = (self.model_path, self.preprocessor_path)
        
        # Create a mock for ModelMonitor instance
        mock_monitor_instance = MagicMock()
        mock_model_monitor.return_value = mock_monitor_instance
        
        # Call the function
        result = run_pipeline(
            data_path=self.test_data_path,
            output_dir=self.test_output_dir,
            n_estimators=100,
            max_depth=None,
            max_features='sqrt',
            test_size=0.2
        )
        
        # Assert that all functions were called with the correct parameters
        mock_load_data.assert_called_once_with(self.test_data_path)
        mock_prepare_data.assert_called_once_with(self.sample_df)
        mock_preprocess_data.assert_called_once_with(self.sample_df)
        
        # Check split_data parameters
        mock_split_data.assert_called_once()
        args, kwargs = mock_split_data.call_args
        self.assertEqual(kwargs['test_size'], 0.2)
        
        # Check train_model parameters
        mock_train_model.assert_called_once()
        args, kwargs = mock_train_model.call_args
        self.assertEqual(args[0].tolist(), self.X_train.tolist())
        self.assertEqual(args[1].tolist(), self.y_train.tolist())
        self.assertEqual(kwargs['model_params'], {
            'n_estimators': 100,
            'max_depth': None,
            'max_features': 'sqrt'
        })
        
        # Check evaluate_model parameters
        mock_evaluate_model.assert_called_once_with(mock_model, self.X_test, self.y_test)
        
        # Check monitoring
        mock_monitor_instance.log_training_metrics.assert_called_once_with(
            metrics=self.eval_results,
            params={
                'n_estimators': 100,
                'max_depth': None,
                'max_features': 'sqrt',
                'test_size': 0.2
            }
        )
        
        # Check save_model parameters
        mock_save_model.assert_called_once_with(mock_model, mock_preprocessor, self.test_output_dir)
        
        # Check return value
        self.assertEqual(result, {
            'model_path': self.model_path,
            'preprocessor_path': self.preprocessor_path,
            'evaluation_results': self.eval_results
        })
        
        # Check if CSV files were saved
        expected_files = [
            os.path.join(self.test_output_dir, "intermediates", "01_raw_data.csv"),
            os.path.join(self.test_output_dir, "intermediates", "02_prepared_data.csv"),
            os.path.join(self.test_output_dir, "intermediates", "03_preprocessed_data.csv")
        ]
        for file_path in expected_files:
            self.assertTrue(os.path.exists(file_path))
    
    @patch('sentiment_analysis_model.run_training_pipeline.os.path.exists')
    @patch('sentiment_analysis_model.run_training_pipeline.os.makedirs')
    def test_output_directory_creation(self, mock_makedirs, mock_exists):
        """Test that the output directory is created if it doesn't exist"""
        
        # Configure mocks to simulate directory not existing
        mock_exists.return_value = False
        
        # Need to patch all other functions to prevent actual execution
        with patch('sentiment_analysis_model.run_training_pipeline.ModelMonitor'), \
            patch('sentiment_analysis_model.run_training_pipeline.load_data', return_value=self.sample_df), \
            patch('sentiment_analysis_model.run_training_pipeline.prepare_data', return_value=self.sample_df), \
            patch('sentiment_analysis_model.run_training_pipeline.preprocess_data', return_value=(self.processed_df, MagicMock())), \
            patch('sentiment_analysis_model.run_training_pipeline.split_data', return_value=(self.X_train, self.X_test, self.y_train, self.y_test)), \
            patch('sentiment_analysis_model.run_training_pipeline.train_model', return_value=MagicMock()), \
            patch('sentiment_analysis_model.run_training_pipeline.evaluate_model', return_value=self.eval_results), \
            patch('sentiment_analysis_model.run_training_pipeline.save_model', return_value=(self.model_path, self.preprocessor_path)):
            
            
            # Call the function
            run_pipeline(self.test_data_path, self.test_output_dir)
            
            # Check that makedirs was called with the expected path
            intermediates_dir = os.path.join(self.test_output_dir, "intermediates")
            mock_makedirs.assert_called_with(intermediates_dir)
    
  

if __name__ == '__main__':
    unittest.main()
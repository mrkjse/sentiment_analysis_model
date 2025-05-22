import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from sentiment_analysis_model.text_preprocessor import TextPreprocessor, preprocess_data


class TestTextPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_preprocess_combined(self):

        text = "ThIS is a TexT with 123 numbers and UPPERCASE letters..."
        processed = self.preprocessor.preprocess(text)
        self.assertEqual(processed, "text numbers uppercase letters")

    def test_preprocess_data_function(self):

        # Create a sample dataframe
        data = {
            'review_text': ['This is a good product!']
        }
        df = pd.DataFrame(data)
        
        result_df, returned_preprocessor = preprocess_data(df, self.preprocessor)
        
        self.assertEqual(returned_preprocessor, self.preprocessor)
        self.assertIn('processed_review', result_df.columns)
        
        self.assertEqual(result_df['processed_review'][0], "good product")

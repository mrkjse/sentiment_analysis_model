import os
import json
import logging
import time
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ModelMonitor:
    
    def __init__(self, output_dir="model_monitoring"):
        self.output_dir = output_dir
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        
        # File to store metrics and predictions
        self.metrics_file = os.path.join(output_dir, "metrics_history.json")
        self.inference_file = os.path.join(output_dir, "inference_log.json")
        
        # Load existing metrics if available
        self.metrics_history = []
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
            except:
                self.metrics_history = []
        
        # Load existing inference logs if available
        self.inference_logs = []
        if os.path.exists(self.inference_file):
            try:
                with open(self.inference_file, 'r') as f:
                    self.inference_logs = json.load(f)
            except:
                self.inference_logs = []
                
        logger.info(f"Model monitor initialized in {output_dir}")
    

    def log_training_metrics(self, metrics, params=None):
        """
        Log training metrics for monitoring.

        Parameters
        ----------
        self : 
            The ModelMonitor instance.
        
        metrics : dict
            The dictionary of metrics and their corresponding scores.
        
        paarams : string
            The dictionary of the hyperparameters of the model.


        Returns
        -------
        metrics_entry: dict
            The dictionary log containing the information provided.

        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        metrics_entry = {
            "timestamp": timestamp,
            "metrics": metrics,
            "parameters": params or {}
        }
        
        self.metrics_history.append(metrics_entry)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
        logger.info(f"Logged training metrics: {metrics}")
        
        return metrics_entry
    

    def log_prediction(self, text, prediction, confidence):
        """
        Log a prediction for monitoring.

        Parameters
        ----------
        self : 
            The ModelMonitor instance.
        
        text : string
            The text being predicted.
        
        prediction : string
            The predicted label (Negative, Neutral, Positive).

        confidence : dict
            A dictionary of the label confidence scores.


        Returns
        -------
        inference_entry: dict
            The dictionary log containing the information provided.

        """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log some info from inference
        inference_entry = {
            "timestamp": timestamp,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": prediction,
            "confidence": confidence
        }
        
        self.inference_logs.append(inference_entry)

        # Save to file
        with open(self.inference_file, 'w') as f:
            json.dump(self.inference_logs, f, indent=2)
        
        return inference_entry
   
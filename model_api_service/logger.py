import os
import numpy as np
import json
import time
from datetime import datetime
from collections import deque

class APILogger:
    
    def __init__(self, log_dir="api_logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "api_requests.json")
        self.stats_file = os.path.join(log_dir, "basic_stats.json")
        
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Keep recent requests in memory
        self.recent_requests = deque(maxlen=100)  # Last 100 requests
        self.response_times = []  # Store response times
        
        print(f"Simple logger initialized in {log_dir}")

         # Load recent requests from file if it exists
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    all_logs = json.load(f)
                    # Take the last 100 entries
                    recent_logs = all_logs[-100:] if len(all_logs) > 100 else all_logs
                    self.recent_requests.extend(recent_logs)
                    # Also restore response times
                    self.response_times = [log['response_time_ms'] for log in recent_logs]
            except Exception as e:
                print(f"Error loading existing logs: {e}")
    
    def log_request(self, request_text, prediction, confidence, response_time):
        """Log an API request with its prediction and response time."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "text": request_text[:50] + "..." if len(request_text) > 50 else request_text,
            "prediction": prediction,
            "response_time_ms": response_time * 1000  # Convert to milliseconds
        }
        
        self.recent_requests.append(log_entry)
        self.response_times.append(response_time * 1000)  # Store in ms
        
        # Keep response_times list from growing too large
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # Save to log file
        try:
            # Load existing logs if file exists
            existing_logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    existing_logs = json.load(f)
            
            # Add new log
            updated_logs = existing_logs + [log_entry]
            
            with open(self.log_file, 'w') as f:
                json.dump(updated_logs, f, indent=2)
            
            # Update basic stats regarding response times and latency
            self._update_stats()
            
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def _update_stats(self):
        """Update simple statistics about API usage."""
        try:
            # Count sentiment types
            sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
            for req in self.recent_requests:
                sentiment = req.get("prediction")
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
            
            # Calculate average response time
            avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            
            # Create stats object
            stats = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_requests": len(self.recent_requests),
                "sentiment_counts": sentiment_counts,
                "avg_response_time_ms": avg_time,
                "p99_response_time_ms": np.percentile(self.response_times, 99) if self.response_times else 0
            }
            
            # Save to file
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            print(f"Error updating stats: {e}")
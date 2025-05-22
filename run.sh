# This is a unified bash script that will run the whole app.
# 
# Steps
# 1. Run the training pipeline and produce the model
# 2. Start the API server


# Run the training container first
echo "Starting training container..."
docker compose up training

# Check if the training pipeline was completed successfully
if [ $? -eq 0 ]; then
  echo "Training completed successfully."
  echo "Starting API service..."
  
  # Start the API service
  docker compose up -d api
  echo "API service is running on http://localhost:8000"
else
  echo "Training failed. Check logs for details."
  exit 1
fi
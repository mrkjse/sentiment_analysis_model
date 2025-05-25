# This is a unified bash script that will run the whole app.
# 
# Steps
# 1. Run the training pipeline and produce the model
# 2. Start the API server

# Stop any existing containers
echo "Stopping any existing containers..."
docker compose down

# Build and run the training container first
echo "Building and starting training container..."
echo "This may take a few minutes if images need to be rebuilt..."
docker compose up --build --force-recreate training

# Check if the training pipeline was completed successfully
if [ $? -eq 0 ]; then
  echo "Training completed successfully."
  echo "Starting API service..."
  
  # Start the API service
  docker compose up -d --build --force-recreate api
  echo "API service is running on http://localhost:8000"
else
  echo "Training failed. Check logs for details."
  exit 1
fi
```
eval $(poetry env activate)

python -m sentiment_analysis_model.run_training_pipeline --data data/reviews.json --output-dir out/

python -m sentiment_analysis_model.run_prediction --model out/model.joblib --preprocessor out/preprocessor.joblib --text 'the book was okay'  

python -m model_api_service.main

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "the book was okay"}'


curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "reading it felt absolutely terrible"}'

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "it was extremely gratifying!!! I would want to read it again and again!!"}'

curl http://localhost:8000/health

# Run just the training
docker-compose up --build training

# Run just the API (assuming training has already been done)
docker-compose up --build api

```
```
eval $(poetry env activate)

python -m sentiment_analysis_model.run_training_pipeline --data data/reviews.json --output-dir out/

python -m sentiment_analysis_model.run_prediction --model out/model.joblib --preprocessor out/preprocessor.joblib --text 'the book was okay'  

python -m model_api_service.main
```

## FastAPI commands

```
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

```

## Docker commands

```
docker-compose up --build

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "the book was okay"}'
```

### Advantages of This Approach:

- More Control: You decide when to run the training process.
- Flexibility: You can run the training with different parameters without rebuilding the container.
- Reusability: You can use the same training container for different training runs.
- Debugging: It's easier to debug issues when running commands manually.
```
cd src
rm -rf .venv
poetry env use python3.10
poetry config virtualenvs.in-project true --local && \
poetry install
eval $(poetry env activate)

poetry run python -m sentiment_analysis_model.run_training_pipeline --data data/reviews.json --output-dir out/

poetry run python -m sentiment_analysis_model.run_prediction --model out/model.joblib --preprocessor out/preprocessor.joblib --text 'the book was okay'  

poetry run python -m model_api_service.main
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

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "the book was so so"}'


curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "i love the story so much!"}'

curl http://localhost:8000/health

curl http://localhost:8000/stats

```

## Docker commands

```
cd src
docker-compose up --build

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "the book was okay"}'


chmod +x src/run.sh
./run.sh

docker ps -a
docker stop aeff01f3924f <container-id>
docker stop bfb57a9b2896 <container-id>


docker start bfb57a9b2896

docker build --no-cache -t training:latest -f Dockerfile.training .

```

### Advantages of This Approach:

- More Control: You decide when to run the training process.
- Flexibility: You can run the training with different parameters without rebuilding the container.
- Reusability: You can use the same training container for different training runs.
- Debugging: It's easier to debug issues when running commands manually.
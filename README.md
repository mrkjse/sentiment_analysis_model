# Amazon Book Review Sentiment Analyser

This is a classifier that categorises review sentences into positive, neutral, and negative. The model is also being served via a FastAPI Inference service that allows a text to be classified with its corresponding confidences passed as well. 

## Running Locally

The solution is Dockerised. There are two Docker containers provided:

- one for the training and prediction service
- another for the API service (fastAPI)

To run this in your local, simply run the bash script `run.sh`.

```bash
# Make sure you are in the root folder and that your Docker daemon is running
./run.sh
```

If you do not have Docker, you may opt to set up the Python environment (Python 3.10.15) via `poetry` on your local and run the scripts.

```bash
pip install poetry
poetry env use python3.10

# Configure to create the virtual environment on the local folder
poetry config virtualenvs.in-project true --local && \

# Create the Python virtual environment
poetry install

# Run the training pipeline
poetry run python -m sentiment_analysis_model.run_training_pipeline --data data/reviews.json --output-dir out/

# Run the prediction pipeline (this will be available as an API as well)
poetry run python -m sentiment_analysis_model.run_prediction --model out/model.joblib --preprocessor out/preprocessor.joblib --text 'the book was okay' 

# Run the API inference service
poetry run python -m model_api_service.main
```

If successful, you should observe the following logs from the **training pipeline**:

```bash
# Sample training pipeline result
training-1  | 2025-05-22 08:07:07,059 - sentiment_analysis_model.model_monitor - INFO - Model monitor initialized in /app/out/monitoring
training-1  | 2025-05-22 08:07:07,059 - __main__ - INFO - Step 1: Loading data
training-1  | 2025-05-22 08:07:07,489 - __main__ - INFO - Step 2: Preparing data
training-1  | 2025-05-22 08:07:07,705 - __main__ - INFO - Step 3: Preprocessing text
training-1  | 2025-05-22 08:07:07,705 - sentiment_analysis_model.utils - INFO - Starting preprocess_data
training-1  | [nltk_data] Downloading package stopwords to /root/nltk_data...
training-1  | [nltk_data]   Package stopwords is already up-to-date!
training-1  | [nltk_data] Downloading package punkt_tab to /root/nltk_data...
training-1  | [nltk_data]   Package punkt_tab is already up-to-date!
training-1  | 2025-05-22 08:07:10,748 - sentiment_analysis_model.utils - INFO - Completed preprocess_data in 3.04 sec
training-1  | 2025-05-22 08:07:11,061 - __main__ - INFO - Step 4: Splitting data
training-1  | 2025-05-22 08:07:11,062 - sentiment_analysis_model.model_trainer - INFO - Splitting the dataset...
training-1  | 2025-05-22 08:07:11,066 - __main__ - INFO - Step 5: Training model
training-1  | 2025-05-22 08:07:11,066 - sentiment_analysis_model.utils - INFO - Starting train_model
training-1  | 2025-05-22 08:07:11,066 - sentiment_analysis_model.model_trainer - INFO - Creating TF-IDF vectorizer and RandomForest model...
training-1  | 2025-05-22 08:07:11,066 - sentiment_analysis_model.model_trainer - INFO - Training basic RandomForestClassifier...
training-1  | 2025-05-22 08:07:13,988 - sentiment_analysis_model.utils - INFO - Completed train_model in 2.92 sec
training-1  | 2025-05-22 08:07:13,988 - __main__ - INFO - Step 6: Evaluating model
training-1  | 2025-05-22 08:07:13,988 - sentiment_analysis_model.model_evaluator - INFO - Evaluating model...
training-1  | 2025-05-22 08:07:14,119 - sentiment_analysis_model.model_evaluator - INFO - Model accuracy: 0.6915
training-1  | 2025-05-22 08:07:14,122 - sentiment_analysis_model.model_evaluator - INFO - Classification report:
training-1  |               precision    recall  f1-score   support
training-1  | 
training-1  |            0       0.71      0.80      0.75       823
training-1  |            1       0.73      0.14      0.23       388
training-1  |            2       0.67      0.85      0.75       789
training-1  | 
training-1  |     accuracy                           0.69      2000
training-1  |    macro avg       0.70      0.60      0.58      2000
training-1  | weighted avg       0.70      0.69      0.65      2000
training-1  | 
training-1  | 2025-05-22 08:07:14,123 - sentiment_analysis_model.model_evaluator - INFO - Confusion matrix:
training-1  | [[655  15 153]
training-1  |  [160  54 174]
training-1  |  [110   5 674]]
training-1  | 2025-05-22 08:07:14,124 - sentiment_analysis_model.model_monitor - INFO - Logged training metrics: {'accuracy': 0.6915, 'classification_report': {'0': {'precision': 0.7081081081081081, 'recall': 0.795868772782503, 'f1-score': 0.7494279176201373, 'support': 823.0}, '1': {'precision': 0.7297297297297297, 'recall': 0.13917525773195877, 'f1-score': 0.23376623376623376, 'support': 388.0}, '2': {'precision': 0.6733266733266733, 'recall': 0.8542458808618505, 'f1-score': 0.753072625698324, 'support': 789.0}, 'accuracy': 0.6915, 'macro avg': {'precision': 0.7037215037215038, 'recall': 0.5964299704587708, 'f1-score': 0.5787555923615649, 'support': 2000.0}, 'weighted avg': {'precision': 0.6985814266814266, 'recall': 0.6915, 'f1-score': 0.6508273882893246, 'support': 2000.0}}, 'confusion_matrix': [[655, 15, 153], [160, 54, 174], [110, 5, 674]]}
training-1  | 2025-05-22 08:07:14,124 - __main__ - INFO - Step 7: Saving model
training-1  | 2025-05-22 08:07:14,398 - sentiment_analysis_model.model_trainer - INFO - Model saved to /app/out/model.joblib
training-1  | 2025-05-22 08:07:14,403 - sentiment_analysis_model.model_trainer - INFO - Preprocessor saved to /app/out/preprocessor.joblib
training-1  | 2025-05-22 08:07:14,403 - __main__ - INFO - Pipeline completed successfully!
training-1  | 2025-05-22 08:07:14,403 - __main__ - INFO - Model saved to: /app/out/model.joblib
training-1  | 2025-05-22 08:07:14,403 - __main__ - INFO - Model accuracy: 0.6915

```

You should also find some model monitoring logs in `out/monitoring/metrics_history.json`:

```json
{
    "timestamp": "2025-05-21 11:28:17",
    "metrics": {
      "accuracy": 0.6905,
      "classification_report": {
        "0": {
          "precision": 0.7166301969365426,
          "recall": 0.795868772782503,
          "f1-score": 0.7541738629821532,
          "support": 823.0
        },
        "1": {
          "precision": 0.6666666666666666,
          "recall": 0.10309278350515463,
          "f1-score": 0.17857142857142858,
          "support": 388.0
        },
        "2": {
          "precision": 0.6686159844054581,
          "recall": 0.8694550063371356,
          "f1-score": 0.7559228650137741,
          "support": 789.0
        },
        "accuracy": 0.6905,
        "macro avg": {
          "precision": 0.6839709493362225,
          "recall": 0.5894721875415977,
          "f1-score": 0.5628893855224519,
          "support": 2000.0
        },
        "weighted avg": {
          "precision": 0.6879956652206738,
          "recall": 0.6905,
          "f1-score": 0.6431969720079471,
          "support": 2000.0
        }
      },
      "confusion_matrix": [
        [
          655,
          17,
          151
        ],
        [
          159,
          40,
          189
        ],
        [
          100,
          3,
          686
        ]
      ]
    },
    "parameters": {
      "n_estimators": 100,
      "max_depth": null,
      "max_features": "sqrt",
      "test_size": 0.2
    }
  },
```
For the prediction pipeline, you should observe the following logs:

```bash
2025-05-22 12:05:17,325 - sentiment_analysis_model.model_monitor - INFO - Model monitor initialized in out/monitoring
Review: the book was okay
Sentiment: Neutral
Confidence scores:
  Negative: 0.2073
  Neutral: 0.7727
  Positive: 0.0200
```
You should also find some inference logs via `out/monitoring/inference_log.json`:

```json
  {
    "timestamp": "2025-05-21 11:28:46",
    "text": "the book was okay",
    "prediction": "Neutral",
    "confidence": {
      "Negative": 0.35669047619047617,
      "Neutral": 0.5961666666666666,
      "Positive": 0.047142857142857146
    }
  }
```

If the fastAPI service is initiated successfully, you should find the following logs:

```bash
INFO:     Will watch for changes in these directories: ['/app']

INFO:     Uvicorn running on http://0.0.0.0:8000⁠ (Press CTRL+C to quit)
```

The following logs are available in the server side to keep track of the API service:

```bash
INFO:     192.168.65.1:41052 - "POST /predict HTTP/1.1" 200 OK

{'sentiment': 'Negative', 'confidence': {'Negative': 0.7760943313811863, 'Neutral': 0.09809587227165027, 'Positive': 0.12580979634716324}}

0.06581521034240723

INFO:     192.168.65.1:61341 - "POST /predict HTTP/1.1" 200 OK

INFO:     192.168.65.1:28390 - "GET /stats HTTP/1.1" 200 OK

2025-05-21 14:03:22,983 - sentiment_analysis_model.utils - INFO - Looking for basic_stats.json in {api_logger.log_dir}

{'sentiment': 'Negative', 'confidence': {'Negative': 0.9742071912761567, 'Neutral': 0.012426318891836131, 'Positive': 0.013366489832007073}}

0.056195735931396484
```

## Sentiment Analyser Model

### Dataset

The dataset used is a sample of 10,000 books from Amazon. 

The training data is provided in JSON Lines format, with each line containing a valid JSON object with the following fields:
- `text`: The review text content (required)
- `rating`: Numerical rating (typically 1-5) (required)
- `title` (optional): The title of the review
- Additional fields are allowed and are preserved in the pipeline

Example:
```json
{"rating": 1.0, "title": "For beginner Antique Jewellery Collectors Only!", "text": "Nothing new for a collector of Antique Jewellery! Pretty much all the pieces on this book have been in other Jewelry books!!!", "asin": "1788841581", "user_id": "AGBUJDDLRJIUJFTKPABJT6CJTHRQ", "verified_purchase": true}
{"rating": 5.0, "title": "Excellent Product", "text": "This is a fantastic item that works exactly as described. Would definitely buy again!", "asin": "B00X4WHP5E", "user_id": "A3QK5MLWOJI3BG", "verified_purchase": true}
{"rating": 3.0, "title": null, "text": "Average quality but good value for the price.", "asin": "B07H2VKSS8", "user_id": "AVNRT6QFR73JK", "verified_purchase": false}
```

### Model Used

The model used is a scikit-learn pipeline with TF-IDF and RandomForestClassifier.

### Model monitoring

The following information is logged upon model training:

- `timestamp`: the time when the model was created
- `metrics`: the metrics score of the model when evaluated against the test set (e.g. the accuracy, confusion matrix, precision, recall, f1-score)
- `parameters`: the hyperparameters used for the model


## Sentiment Analyser API Service

I created a slim version of a fastAPI service to serve model predictions via API. 

### API services

#### POST predict

This loads the trained model and performs the inference. The only parameter needed is `text`, which is a collection of text to be classified as negative, neutral, or positive.


```bash
# Sample POST request
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "it was extremely gratifying!!! I would want to read it again and again!!"}'
```
The response includes the predicted sentiment and the corresponding confidences.

``` json
# API response
{"review":"it was extremely gratifying!!! I would want to read it again and again!!","sentiment":"Negative","confidence":{"Negative":0.74,"Neutral":0.06,"Positive":0.2}}
```

#### GET health

This API service retrieves the health of the prediction. Specifically, it checks if the model exists in the model path.

```bash
# Sample GET request
curl http://localhost:8000/health
```

```json
# Curl response
{"status":"healthy"}%  
```

#### GET stats

This relates to monitoring the latency of the service. There logs related to the API service is available on `out/api_logs`. The `api_requests.json` logs all the requests received by the API, while the `basic_stats.json` updates the statistics related to the latency and the responses returned by the service.

When requested, it returns the following information (from `basic_stats.json`):

- `last_updated`: the last time the API received a request
- `total_requests`: the total number of requests received (since the server has been running)
- `sentiment_counts`: the distribution of the predicted sentiments
- `avg_response_time_ms`: the average response time in milliseconds
- `p99_response_time_ms`: the 99th percentile of the response time in milliseconds

```bash
curl http://localhost:8000/stats 
```

```json
{"last_updated":"2025-05-22 11:12:15","total_requests":1,"sentiment_counts":{"Positive":0,"Neutral":0,"Negative":1},"avg_response_time_ms":237.18738555908203,"p99_response_time_ms":237.18738555908203}% 
```

## Unit Tests

### Unit Tests for Sentiment Analyser Model

The following unit tests are created for the sentiment analyser model:

- `test_run_training_pipeline.py` - To test the expected behaviour when running the training pipeline end to end.
- `test_run_sentiment_analyser.py` - To test the expected behaviour when we load and predict the model.
- `test_text_preprocessor.py` - To test whether we preprocess the `review` strings as expected.

To run the unit tests: 

```bash
poetry run python -m pytest
```

Successful results should show:

```bash
============================================================= test session starts ==============================================================
platform darwin -- Python 3.10.15, pytest-8.3.5, pluggy-1.6.0
rootdir: /Users/markjose/sentiment_analysis_model
configfile: pyproject.toml
plugins: anyio-4.9.0, cov-4.1.0
collected 6 items                                                                                                                              

tests/test_run_training_pipeline.py ..                                                                                                   [ 33%]
tests/test_sentiment_analyser.py ..                                                                                                      [ 66%]
tests/test_text_preprocessor.py ..                                                                                                       [100%]

============================================================== 6 passed in 1.53s ===============================================================

```

## Improving the Latency

When I first tested my Inference service, the API was agonisingly slow:

```json
  {
    "timestamp": "2025-05-21 14:14:38",
    "text": "reading it felt absolutely terrible",
    "prediction": "Negative",
    "response_time_ms": 1516.5529251098633
  },
  {
    "timestamp": "2025-05-21 14:17:33",
    "text": "the book was so so",
    "prediction": "Negative",
    "response_time_ms": 1533.0898761749268
  },

```

The things I did to improve the latency are:

- Created a lightweight version of the model (from **RandomisedSearchCV** to **RandomForestClassifier** with pruned down hyperparameters)
- Implemented **lrucache** (Least Recently Used Cache) that helps on memory management
- Streamlined inference text preprocessing (removed lemmatisation of words) without sacrificing model accuracy
- Opted to use to fastAPI (than Flask) for asynchronous support

Now, the basic stats show that the 99p latency has been less than 50ms:

```json
{
  "last_updated": "2025-05-21 20:16:34",
  "total_requests": 4,
  "sentiment_counts": {
    "Positive": 1,
    "Neutral": 1,
    "Negative": 2
  },
  "avg_response_time_ms": 34.45667028427124,
  "p99_response_time_ms": 42.2948431968689
}
```

## Extending this Solution

1. Use a more lightweight (and explore better) models (ie distilled BERT or XGBoost)
2. Improve preprocessing (find nltk alternative)
3. Introduce threading and parallelism on some operations (eg I/O ops; but I still have to look into this...)
4. Introduce in-memory caching (still have to look into this...)
5. Store the training artefacts (models, logs) in a PostgreSQL database or in a Cloud bucket
6. Implement MLflow for a more comprehensive model tracking 
7. Serve the API service and the training pipeline as a Cloud Function (or in a Cloud Kubernetes Engine)
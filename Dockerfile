# Stage 1: Training
FROM python:3.10-slim AS training

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir poetry psutil && \
    poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root

# Download NLTK data - this is the key addition
RUN poetry run python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt_tab');"


# Run training during the build process
RUN poetry run python -m sentiment_analysis_model.run_training_pipeline --data data/reviews.json --output-dir out/

# Stage 2: API Service
FROM python:3.10-slim

WORKDIR /app

# Install API dependencies
COPY --from=training /app/poetry.lock /app/pyproject.toml ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --only main --no-root

# Copy code and trained models
COPY --from=training /app/model_api_service ./model_api_service/
COPY --from=training /app/sentiment_analysis_model ./sentiment_analysis_model/
COPY --from=training /app/out ./out/

ENV MODEL_PATH=/app/out/model.joblib
ENV PREPROCESSOR_PATH=/app/out/preprocessor.joblib

EXPOSE 8000

# Download NLTK data - this is the key addition
RUN poetry run python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt_tab');"

# Run the API service
ENTRYPOINT ["poetry", "run", "python", "-m", "model_api_service.main"]
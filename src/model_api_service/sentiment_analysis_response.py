from pydantic import BaseModel

class SentimentResponse(BaseModel):
    review: str
    sentiment: str
    confidence: dict
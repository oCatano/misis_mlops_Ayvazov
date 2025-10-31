from typing import List

from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class BatchTextRequest(BaseModel):
    texts: List[str]


class PredictionResponse(BaseModel):
    text: str
    is_toxic: int
    status: str = "success"


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    status: str = "success"


class ModelInfoResponse(BaseModel):
    model_name: str
    model_size: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

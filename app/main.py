from fastapi import FastAPI, HTTPException

from app import model
from app.schemas import (
    BatchPredictionResponse,
    BatchTextRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TextRequest,
)

app = FastAPI(title="Lab_1 Toxicity Classification API")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", model_loaded=model.model is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(text: TextRequest):
    if len(text.text) == 0:
        raise HTTPException(status_code=400, detail="empty text")

    try:
        pred_class = model.evaluate_text(text.text)
        return {"text": text.text, "is_toxic": pred_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed predict {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(texts: BatchTextRequest):
    if len(texts.texts) == 0:
        raise HTTPException(status_code=400, detail="empty texts")

    for text in texts.texts:
        if len(text) == 0:
            raise HTTPException(status_code=400, detail="empty text")

    try:
        pred_classes = model.evaluate_batch(texts.texts)
        response_list = []
        for text, label in zip(texts.texts, pred_classes):
            response_list.append(PredictionResponse(text=text, is_toxic=label))

        return {"predictions": response_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed predict {str(e)}")


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    memory_mb = (
        sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1024**2
    )
    return {
        "model_name": "s-nlp/russian_toxicity_classifier",
        "model_size": f"{memory_mb:.1f} MB",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

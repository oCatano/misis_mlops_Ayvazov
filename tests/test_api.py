import sys

sys.path.insert(0, "/app")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_200_and_expected_fields():
    """Тест что /health возвращает 200 и ожидаемую структуру ответа"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


def test_predict_endpoint_returns_valid_structure_and_class_range():
    """Тест что /predict возвращает корректную структуру и класс в диапазоне 0-1"""
    response = client.post(
        "/predict", json={"text": "Потрясающая игра, спасибо большое!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] in [0, 1]
    assert data["status"] == "success"


def test_predict_batch_multiple_texts():
    """Тест что /predict_batch обрабатывает несколько текстов и возвращает правильное количество предсказаний"""
    texts = ["Потрясающая игра, спасибо большое!", "Полный отстой, от игры воняет"]
    response = client.post("/predict_batch", json={"texts": texts})

    if response.status_code != 200:
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        print(f"Headers: {response.headers}")

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(texts)

    # Проверяем что каждый элемент имеет ожидаемую структуру
    for prediction in data["predictions"]:
        assert "text" in prediction
        assert "is_toxic" in prediction
        assert prediction["is_toxic"] in [0, 1]


def test_model_info_returns_expected_model_metadata():
    """Тест что /model_info возвращает информацию о модели"""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    # Проверяем основные поля метаданных модели
    assert "model_name" in data
    assert "model_size" in data


def test_predict_endpoint_handles_edge_cases():
    """Тест что /predict корректно обрабатывает пограничные случаи"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400

    response = client.post(
        "/predict", json={"text": "текст с спецсимволами !@#$%^&*()"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_toxic"] in [0, 1]

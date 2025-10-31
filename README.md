# misis_mlops_Ayvazov

## Билд
docker build -t toxic .

## Тесты
docker run -it toxic python -m pytest tests/ -v

## Запуск сервера
docker run -p 8001:8001 toxic

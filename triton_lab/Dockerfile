FROM nvcr.io/nvidia/tritonserver:23.12-py3

WORKDIR /app

EXPOSE 8083 8001 8002

COPY model_repository /models

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

CMD ["tritonserver", "--model-repository=/models", "--http-port=8083"]

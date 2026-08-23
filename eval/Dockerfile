FROM python:3.11-slim
WORKDIR /app
COPY eval/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY eval/mock_backend.py .
EXPOSE 9000
CMD ["python", "mock_backend.py"]

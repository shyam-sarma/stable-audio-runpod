FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg git libsndfile1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TRANSFORMERS_CACHE=/app/hf_cache

EXPOSE 7860

CMD ["python", "app.py"]

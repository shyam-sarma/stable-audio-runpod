FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y ffmpeg git
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose the port used by FastAPI
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates --fresh

COPY requirements.txt .
RUN pip install --upgrade pip certifi
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --no-cache-dir --retries 5 --timeout 180
RUN pip install --no-cache-dir --retries 5 --timeout 180 -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONHASHSEED=42

CMD ["python", "-m", "madewithml.main"]

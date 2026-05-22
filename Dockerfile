FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --retries 5 --timeout 180 --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONHASHSEED=42

CMD ["python", "-m", "madewithml.main"]

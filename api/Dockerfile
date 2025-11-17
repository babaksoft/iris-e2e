
FROM python:3.12-slim

WORKDIR /model

COPY api/model .

WORKDIR /app

COPY api/requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY api/app /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

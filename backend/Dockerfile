FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu torchvision==0.21.0+cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

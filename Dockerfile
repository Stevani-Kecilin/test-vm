FROM python:3.9-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git nano build-essential -y

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
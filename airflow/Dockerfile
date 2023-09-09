FROM python:3.9-slim
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://get.docker.com | sh
RUN pip install --no-cache-dir -r requirements.txt
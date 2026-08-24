FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential
WORKDIR /app

COPY Requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=1000 -r Requirements.txt

COPY . .

CMD ["python", "--version"]

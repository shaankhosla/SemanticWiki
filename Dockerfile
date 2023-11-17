FROM python:3.11-slim-bookworm

RUN mkdir /model_cache

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/model_cache')"

WORKDIR /app

COPY . /app

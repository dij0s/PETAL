# build stage
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y curl

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# install ollama and
# pull required models
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama serve & sleep 5 ; ollama pull llama3.2:3b ; ollama pull nomic-embed-text:v1.5

# copy application code and dependency files
WORKDIR /app
COPY . /app
RUN chmod +x start.sh

# production stage
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl

# install uv and ollama
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN curl -fsSL https://ollama.com/install.sh | sh

# copy ollama models from builder
COPY --from=builder /root/.ollama /root/.ollama

# copy app code and venv from builder
WORKDIR /app
COPY --from=builder /app /app

# expose the port the server runs on
EXPOSE 8000

# start the app
CMD ["./start.sh"]

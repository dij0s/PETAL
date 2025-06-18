# build stage
FROM python:3.11-slim AS builder

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# copy application code and dependency files
WORKDIR /app
COPY . /app

# production stage
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# copy app code and venv from builder
WORKDIR /app
COPY --from=builder /app /app

# expose the port the server runs on
EXPOSE 8000

# start the app
CMD ["uv", "run", "--no-sync", "fastapi", "run", "src/server.py", "--port", "8000", "--host", "0.0.0.0"]

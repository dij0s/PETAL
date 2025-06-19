# build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# copy dependency files
WORKDIR /app
COPY ./pyproject.toml ./uv.lock ./.python-version /app/

# install dependencies
RUN uv sync --no-dev --no-sources

# production stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# copy virtual environment
COPY --from=builder /app/.venv /app/.venv

# copy application
COPY . /app

ENV PATH="/app/.venv/bin:$PATH"

# expose the port the server runs on
EXPOSE 8000

# reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# start the app
CMD ["fastapi", "run", "src/server.py", "--port", "8000", "--host", "0.0.0.0"]

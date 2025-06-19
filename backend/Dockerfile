FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# copy application code and dependency files
WORKDIR /app
COPY . /app

# install dependencies
RUN uv sync --no-dev --no-sources

ENV PATH="/app/.venv/bin:$PATH"

# expose the port the server runs on
EXPOSE 8000

# reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# start the app
CMD ["fastapi", "run", "src/server.py", "--port", "8000", "--host", "0.0.0.0"]

#!/usr/bin/env bash
set -e # quit asap

# start ollama
ollama serve &
sleep 5

# start fastapi server
uv run fastapi run src/server.py --port 8000 --host 0.0.0.0

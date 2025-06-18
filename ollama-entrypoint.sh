#!/bin/bash

echo "Starting Ollama server..."
ollama serve &
SERVE_PID=$!

echo "Waiting for Ollama server to be active..."
while ! ollama list | grep -q 'NAME'; do
  sleep 3
done
# pull models stated in
# environment variables
models_to_download=()
for var in $(env | grep '^OLLAMA_MODEL_' | cut -d= -f2 | sort | uniq); do
  if [ -n "$var" ]; then
    models_to_download+=("$var")
  fi
done
for model in "${models_to_download[@]}"; do
  echo "Pulling model: ${model}"
  ollama pull "${model}"
done

# await end of process
wait $SERVE_PID

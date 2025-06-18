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
echo "Pulling reasoning model: ${REASONING_MODEL}"
ollama pull "${REASONING_MODEL}"
echo "Pulling embedding model: ${EMBEDDING_MODEL}"
ollama pull "${EMBEDDING_MODEL}"
# await end of process
wait $SERVE_PID

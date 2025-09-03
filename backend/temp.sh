#!/bin/bash

# === CONFIGURATION ===
PID=$1  # Pass the PID as the first argument
ENV_FILE="../.env.dev"
OLD_MODEL="REASONING_MODEL=qwen3:1.7b"
NEW_MODEL="REASONING_MODEL=qwen3:8b"

# === CHECK FOR PID ===
if [ -z "$PID" ]; then
    echo "Usage: $0 <PID>"
    exit 1
fi

# === WAIT FOR PID TO FINISH ===
echo "Waiting for process with PID $PID to finish..."
while kill -0 "$PID" 2>/dev/null; do
    sleep 1
done
echo "Process $PID ended."

# === MODIFY ENV FILE ===
if [ -f "$ENV_FILE" ]; then
    echo "Updating reasoning model in $ENV_FILE..."
    sed -i "s|$OLD_MODEL|$NEW_MODEL|" "$ENV_FILE"
else
    echo "Error: Env file $ENV_FILE not found."
    exit 1
fi

# === RUN BENCHMARK ===
echo "Starting benchmark..."
uv run --env-file "$ENV_FILE" src/benchmark.py --iterations 10 --template benchmarking/sion_qwen_benchmark.json --file benchmarking/sion_testcase.txt

# === SHUTDOWN ===
echo "Benchmark completed. Shutting down..."


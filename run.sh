#!/bin/bash

# default to development
ENVIRONMENT=${1:-dev}
if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "prod" ]]; then
    echo "ERROR: Environment must be either 'dev' or 'prod'"
    echo "Usage: ./start.sh [dev|prod] [docker-compose args]"
    exit 1
fi

shift 1 2>/dev/null || true

set -a
if [[ ! -f ".env.$ENVIRONMENT" ]]; then
    echo "ERROR: .env.$ENVIRONMENT file does not exist."
    exit 1
fi
source .env.$ENVIRONMENT
set +a

# export ENVIRONMENT
# for interpolation in
# compose definition
export ENVIRONMENT
"$@"

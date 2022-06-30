#!/bin/bash

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")"


(cd ../..; ./make.sh -j$(nproc) tools)

export PPL_FANOUT_PORT=${PPL_FANOUT_PORT:-'8080'}
export COMPOSE_PROJECT_NAME=ppl-dev

echo "Dev env is launching on http://localhost:$PPL_FANOUT_PORT. Stop it with Ctrl+C"

docker-compose up --build --remove-orphans

# Remove containers and network, keep volumes
docker-compose down --remove-orphans

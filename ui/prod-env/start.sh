#!/bin/bash

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")"


(cd ../..; ./make.sh -j$(nproc) tools)

docker-compose up --build --remove-orphans --detach

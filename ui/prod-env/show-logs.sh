#!/bin/bash

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")"


docker-compose logs

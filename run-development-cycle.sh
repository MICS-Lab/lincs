#!/bin/bash

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")/"


docker build --build-arg UID=$(id -u) development --tag lincs-development

docker run \
  --rm --interactive --tty \
  --volume "$PWD:/wd" --workdir /wd \
  $gpu_arguments \
  lincs-development \
    python3 development/cycle.py "$@"

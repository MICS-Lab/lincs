#!/bin/bash
# Copyright 2023 Vincent Jacques

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")/"


docker build --build-arg UID=$(id -u) development --tag lincs-development

docker run \
  --rm --interactive --tty \
  --env CCACHE_DIR=/wd/ccache \
  --volume "$PWD:/wd" --workdir /wd \
  --publish "8888:8888" \
  lincs-development bash -c """
set -o errexit

python3 -m pip install --user .

jupyter-lab \
  --no-browser \
  --IdentityProvider.token="" --ip=0.0.0.0
"""

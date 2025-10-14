#!/bin/bash
# Copyright 2023-2024 Vincent Jacques

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")/"


docker build --build-arg UID=$(id -u) development --tag lincs-development

docker run \
  --rm --interactive --tty \
  --env CCACHE_DIR=/wd/ccache \
  --mount type=bind,src=$HOME/.gitconfig,dst=/home/user/.gitconfig,ro \
  --mount type=bind,src=$HOME/.ssh/id_rsa,dst=/home/user/.ssh/id_rsa,ro \
  --mount type=bind,src=$HOME/.ssh/known_hosts,dst=/home/user/.ssh/known_hosts \
  --volume "$PWD:/wd" --workdir /wd \
  lincs-development \
    python3 development/publish.py "$@"

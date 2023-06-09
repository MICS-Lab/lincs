#!/bin/bash
# Copyright 2023 Vincent Jacques

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")/"


docker build --build-arg UID=$(id -u) development --tag lincs-development

docker run \
  --rm --interactive --tty \
  --mount type=bind,src=$HOME/.gitconfig,dst=/home/user/.gitconfig,ro \
  --mount type=bind,src=$HOME/.ssh/id_rsa,dst=/home/user/.ssh/id_rsa,ro \
  --mount type=bind,src=$HOME/.ssh/known_hosts,dst=/home/user/.ssh/known_hosts \
  --mount type=bind,src=$HOME/.pypirc,dst=/home/user/.pypirc,ro \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --mount type=bind,src=$HOME/.docker/config.json,dst=/home/user/.docker/config.json,ro \
  --volume "$PWD:/wd" --workdir /wd \
  lincs-development \
    python3 development/publish.py "$@"

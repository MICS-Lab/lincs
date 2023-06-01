#!/bin/bash
# Copyright 2023 Vincent Jacques

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")/"


# @todo Use the actual GID of the docker group
docker build --build-arg UID=$(id -u) --build-arg DOCKER_GID=121 development --tag lincs-development

docker run \
  --rm --interactive --tty \
  --env HOST_ROOT_DIR=$PWD \
  --env CCACHE_DIR=/wd/ccache \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume "$PWD:/wd" --workdir /wd \
  --gpus all `# @todo Support developping on non-GPU machines` \
  $gpu_arguments \
  lincs-development \
    python3 development/cycle.py "$@"

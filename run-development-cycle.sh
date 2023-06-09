#!/bin/bash
# Copyright 2023 Vincent Jacques

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")/"

forbid_gpu=false
cycle_arguments=()
while [[ "$#" -gt 0 ]]
do
  case $1 in
    --forbid-gpu)
      forbid_gpu=true;;
    *)
      cycle_arguments+=($1);;
  esac
  shift
done

if $forbid_gpu
then
  use_gpu=false
elif docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1
then
  use_gpu=true
else
  use_gpu=false
  echo "*************************************************************************************"
  echo "WARNING: the '--gpus all' option of 'docker run' doesn't seem to work on this system."
  echo "This may be because this system doesn't have an NVidia GPU,"
  echo "or because the NVidia drivers or container runtime are not properly installed."
  echo "In any case, the development cycle will skip all tests that use a GPU."
  echo "You can avoid this warning by using the '--forbid-gpu' option."
  echo "*************************************************************************************"
fi

if $use_gpu
then
  docker_run_gpu_arguments="--gpus all"
else
  docker_run_gpu_arguments=""
  cycle_arguments+=("--forbid-gpu")
fi


docker build --build-arg UID=$(id -u) --build-arg DOCKER_GID=$(getent group docker | cut -d ':' -f 3) development --tag lincs-development

docker run \
  --rm --interactive --tty \
  --env HOST_ROOT_DIR=$PWD \
  --env CCACHE_DIR=/wd/ccache \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume "$PWD:/wd" --workdir /wd \
  $docker_run_gpu_arguments \
  lincs-development \
    python3 development/cycle.py "${cycle_arguments[@]}"

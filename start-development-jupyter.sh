#!/bin/bash
# Copyright 2023 Vincent Jacques

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")/"

forbid_gpu=false
while [[ "$#" -gt 0 ]]
do
  case $1 in
    --forbid-gpu)
      forbid_gpu=true;;
    *)
      exit 1;;
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
  echo "You can avoid this warning by using the '--forbid-gpu' option."
  echo "*************************************************************************************"
fi

if $use_gpu
then
  docker_run_gpu_arguments="--gpus all"
else
  docker_run_gpu_arguments=""
fi


docker build --build-arg UID=$(id -u) development --tag lincs-development

docker run \
  --rm --interactive --tty \
  --env CCACHE_DIR=/wd/ccache \
  --env LINCS_DEV_PYTHON_VERSIONS=3.8 \
  --volume "$PWD:/wd" --workdir /wd \
  $docker_run_gpu_arguments \
  --publish "8888:8888" \
  lincs-development bash -c """
set -o errexit

python3.8 -m pip install --user .

jupyter-lab \
  --no-browser \
  --IdentityProvider.token="" --ip=0.0.0.0
"""

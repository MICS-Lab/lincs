#!/bin/bash

set -o errexit
cd "$(dirname "${BASH_SOURCE[0]}")/"


docker build --build-arg UID=$(id -u) development --tag lincs-development


if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1
then
  gpu_arguments="--gpus all"
else
  echo "************************************************************************"
  echo "** The NVidia Docker runtime does not seem to be properly configured. **"
  echo "** This machine may not even have a GPU. The cycle will proceed,      **"
  echo "** but TESTS USING A GPU WILL BE SKIPPED.                             **"
  echo "************************************************************************"
  gpu_arguments=""
fi


docker run \
  --rm --interactive --tty \
  --volume "$PWD:/wd" --workdir /wd \
  $gpu_arguments \
  lincs-development \
    python3 development/cycle.py "$@"

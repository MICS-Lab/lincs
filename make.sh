#!/usr/bin/env bash
# Copyright 2021-2022 Vincent Jacques

set -o errexit

cd "$(dirname "${BASH_SOURCE[0]}")"


if ! diff -r builder build/builder >/dev/null 2>&1 || ! diff Makefile build/Makefile >/dev/null 2>&1
then
  rm -rf build
  mkdir build
  docker build --tag parallel-preference-learning-builder builder
  cp -r builder build
  cp Makefile build
fi


if test -f build/nvidia-docker-runtime.ok || docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1
then
  touch build/nvidia-docker-runtime.ok
else
  echo "************************************************************************"
  echo "** The NVidia Docker runtime does not seem to be properly configured. **"
  echo "************************************************************************"
  exit 1
fi


docker run \
  --rm --interactive --tty \
  --user $(id -u):$(id -g) `# Avoid creating files as 'root'` \
  --gpus all \
  --network none `# Ensure the repository is self-contained (except for the "docker build" phase)` \
  --volume "$PWD:/wd" --workdir /wd \
  parallel-preference-learning-builder \
    make "$@"

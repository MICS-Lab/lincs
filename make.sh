#!/usr/bin/env bash
# Copyright 2021-2022 Vincent Jacques

set -o errexit

cd "$(dirname "${BASH_SOURCE[0]}")"

# One can export variables PPL_SKIP_* before running this script to skip some parts.
# Useful to spare some time on repeated runs.

# Fail fast if Docker and NVidia's container runtime are not configured
if [[ -z $PPL_SKIP_CHECK_GPU ]]
then
  if ! docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1
  then
    echo "ERROR: Docker cannot use a CUDA-capable GPU. Please check the configuration of NVidia's container runtime"
    exit 1
  fi
fi

if [[ -z $PPL_SKIP_BUILDER ]]
then
  id_before=$(docker image inspect parallel-preference-learning-builder -f '{{.Id}}' 2>/dev/null || echo none)

  docker build builder --tag parallel-preference-learning-builder

  id_after=$(docker image inspect parallel-preference-learning-builder -f '{{.Id}}')

  # Force full rebuild when dependencies change
  if [[ $id_before != $id_after ]]
  then
    rm -rf build
  fi
fi

docker run \
  --rm --interactive --tty \
  --user $(id -u):$(id -g) `# Avoid creating files as 'root'` \
  --gpus all \
  --network none `# Ensure the repository is self-contained (except for the "docker build" phase)` \
  --volume "$PWD:/wd" --workdir /wd \
  parallel-preference-learning-builder \
    make "$@"

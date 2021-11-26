#!/usr/bin/env bash
# Copyright 2021 Vincent Jacques

set -o errexit

cd "$(dirname "${BASH_SOURCE[0]}")"

# Fail fast if Docker and NVidia's container runtime are not configured
if ! docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1
then
  echo "ERROR: Docker cannot use a CUDA-capable GPU. Please check the configuration of NVidia's container runtime"
  exit 1
fi

if [[ -z $PPL_SKIP_BUILDER ]]
then
  docker build builder --tag parallel-preference-learning-builder
fi

test -f core || (echo "No 'core' file. Please check /proc/sys/kernel/core_pattern")

docker run \
  --rm --interactive --tty \
  --user $(id -u):$(id -g) \
  --volume "$PWD:/wd" --workdir /wd \
  parallel-preference-learning-builder \
    bash -c """
set -o errexit

executable=\$(file core | sed \"s/.*from '\(.*\)', real.*/\1/\")

gdb \$executable core
"""

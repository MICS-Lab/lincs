# Copyright 2023 Vincent Jacques

FROM ubuntu:22.04

# System packages
RUN set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      ca-certificates \
      ccache \
      dirmngr \
      docker.io \
      g++ \
      git \
      gpg-agent \
      libboost-python-dev \
      libyaml-cpp-dev \
      python3-dev \
      python3-pip \
      python3-venv \
      software-properties-common \
      ssh \
      wget \
 && apt-get clean

# NVidia packages
RUN set -x \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
 && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      cuda-cudart-dev-12-1 \
      cuda-nvcc-12-1 \
 && apt-get clean

# OR-Tools
# DO NOT UPGRADE to version 9.y: there is a SegFault bug (that I still need to report)
RUN set -x \
 && wget https://github.com/google/or-tools/releases/download/v8.2/or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && tar xf or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/include/* /usr/local/include \
 && cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/lib/*.so /usr/local/lib \
 && rm -r or-tools_Ubuntu-20.04-64bit_v8.2.8710 or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && ldconfig

# MagicEnum
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/Neargye/magic_enum/v0.8.2/include/magic_enum.hpp

# RapidCSV
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/d99kris/rapidcsv/v8.75/src/rapidcsv.h

# Doctest
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h

# Lov-e-cuda
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/jacquev6/lov-e-cuda/13e45bc/lov-e.hpp

# Python packages
RUN pip3 install \
      build \
      semver \
      sphinx \
      sphinx-click \
      twine

# Python packages again, to speed-up development cycle
RUN pip3 install \
      click \
      matplotlib

# User
ARG UID
RUN useradd user --uid $UID --create-home
ARG DOCKER_GID
RUN groupadd hostdocker --gid $DOCKER_GID
RUN adduser user hostdocker
USER user
RUN mkdir -p /home/user/.local/bin
ENV PATH=/home/user/.local/bin:$PATH:/usr/local/cuda-12.1/bin

# CCache
RUN ln -s $(which ccache) /home/user/.local/bin/x86_64-linux-gnu-gcc
RUN ln -s $(which ccache) /home/user/.local/bin/nvcc

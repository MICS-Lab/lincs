# Copyright 2023-2024 Vincent Jacques

# About the choice of Ubuntu 20.04 instead of 22.04: it an acceptable compromise:
# - it's an LTS version as well
# - the deadsnakes PPA supports it
# - NVidia supports the same versions of CUDA as for 22.04
# - its gcc version supports C++17
# - it allows building manylinux_2_31 wheels, which are slightly more widely usable than the manylinux_2_34 wheels produced on 22.04

FROM ubuntu:20.04 AS base

# https://docs.docker.com/engine/reference/builder/#run---mounttypecache
RUN set -x \
 && rm -f /etc/apt/apt.conf.d/docker-clean \
 && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache


FROM base AS downloader

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        ca-certificates \
        wget

WORKDIR /download


FROM downloader AS download-boost

RUN set -x \
 && wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.gz \
 && tar xf boost_*.tar.gz \
 && rm boost_*.tar.gz \
 && mv boost_* boost


FROM downloader AS download-get-pip

RUN set -x \
 && wget https://bootstrap.pypa.io/get-pip.py


FROM downloader AS download-or-tools

# DO NOT UPGRADE to version 9.y: there is a SegFault bug (that I still need to report)
RUN set -x \
 && wget https://github.com/google/or-tools/releases/download/v8.2/or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && tar xf or-tools_*.tar.gz \
 && rm or-tools_*.tar.gz \
 && mv or-tools_* or-tools


FROM downloader AS download-patchelf

RUN set -x \
 && wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz \
 && mkdir patchelf \
 && cd patchelf \
 && tar xf ../patchelf-*.tar.gz \
 && cd .. \
 && rm patchelf-*.tar.gz


FROM base AS final

# System packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      ccache \
      dirmngr \
      docker.io \
      g++ \
      git \
      gpg-agent \
      graphviz \
      jq \
      pandoc \
      python3-dev \
      python3-venv \
      software-properties-common \
      ssh \
      sudo

# Python versions
ENV LINCS_DEV_PYTHON_VERSIONS="3.8 3.9 3.10 3.11 3.12"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=bind,from=download-get-pip,source=/download,target=/download \
    set -x \
 && default_python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") \
 && non_default_python_versions=$(\
      for python_version in $LINCS_DEV_PYTHON_VERSIONS; do echo $python_version; done \
      | grep -v $default_python_version \
    ) \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      $(for python_version in $non_default_python_versions; do echo "python$python_version-dev python$python_version-distutils python$python_version-venv"; done) \
 && for python_version in $non_default_python_versions $default_python_version; do python$python_version /download/get-pip.py || exit 1; done

# Boost
RUN --mount=type=bind,from=download-boost,source=/download,target=/download,readwrite \
    set -x \
 && cd /download/boost \
 && (for version in $LINCS_DEV_PYTHON_VERSIONS; do echo "using python : $version ;"; done) >tools/build/src/user-config.jam \
 && ./bootstrap.sh --with-libraries=python \
 && ./b2 $(for version in $LINCS_DEV_PYTHON_VERSIONS; do echo "python=$version"; done) \
 && cp -r /download/boost/boost /usr/local/include \
 && cp -r /download/boost/stage/lib/* /usr/local/lib

# NVidia packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -x \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
 && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      cuda-cudart-dev-12-1 \
      cuda-nvcc-12-1

# OR-Tools
RUN --mount=type=bind,from=download-or-tools,source=/download,target=/download \
    set -x \
 && cp -r /download/or-tools/include/* /usr/local/include \
 && cp -r /download/or-tools/lib/*.so /usr/local/lib

# Python packages: development and publication tools
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
 && pip3 install \
      auditwheel \
      bash_kernel \
      Chrones \
      gcovr \
      joblib \
      jupyterlab \
      lark \
      myst-parser \
      parver \
      sphinx \
      sphinx-click \
      sphinx-jsonschema \
      sphinxcontrib-details-directive \
      twine \
 && python3 -m bash_kernel.install \
 && for python_version in $LINCS_DEV_PYTHON_VERSIONS; do python$python_version -m pip install \
      build \
    || exit 1; done
# Recent patchelf version for 'auditwheel repair'
RUN --mount=type=bind,from=download-patchelf,source=/download,target=/download \
    set -x \
 && cp /download/patchelf/bin/patchelf /usr/local/bin

# Python packages: runtime dependencies
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    for python_version in $LINCS_DEV_PYTHON_VERSIONS; do python$python_version -m pip install \
      click~=8.1 \
      matplotlib~=3.7 \
      six~=1.16 \
    || exit 1; done

# Finalization
RUN ldconfig

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      gdb

# User
ARG UID
RUN useradd user --uid $UID --create-home
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user
USER user
RUN mkdir -p /home/user/.local/bin
ENV PATH=/home/user/.local/bin:$PATH:/usr/local/cuda-12.1/bin

# Make sure /home/user/.local/lib/python3.8/site-packages is in sys.path
# (useful for development/cycle.py, where 'pip install --user lincs' is run just before 'import lincs')
RUN pip3 install --user noop

# CCache
RUN ln -s $(which ccache) /home/user/.local/bin/x86_64-linux-gnu-gcc
RUN ln -s $(which ccache) /home/user/.local/bin/x86_64-linux-gnu-g++
RUN ln -s $(which ccache) /home/user/.local/bin/nvcc

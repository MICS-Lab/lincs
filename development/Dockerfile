# Copyright 2023 Vincent Jacques

FROM ubuntu:20.04
# About the choice of Ubuntu 20.04 instead of 22.04: it an acceptable compromise:
# - it's an LTS version as well
# - the deadsnakes PPA supports it
# - NVidia supports the same versions of CUDA as for 22.04
# - its gcc version supports C++17
# - it allows building manylinux_2_31 wheels, which are slightly more widely usable than the manylinux_2_34 wheels produced on 22.04

# @todo Clean-up intermediate files (tarballs, source and build directories, etc.)
# @todo Use Docker build stages to avoid re-downloading tarballs (but keep it compatible with the clean-up of intermediate files)

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
      graphviz \
      libyaml-cpp-dev \
      python3-dev \
      python3-venv \
      software-properties-common \
      ssh \
      sudo \
      wget \
 && apt-get clean

# Python versions
ENV LINCS_DEV_PYTHON_VERSIONS="3.7 3.8 3.9 3.10 3.11"
RUN set -x \
 && default_python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") \
 && non_default_python_versions=$(\
      for python_version in $LINCS_DEV_PYTHON_VERSIONS; do echo $python_version; done \
      | grep -v $default_python_version \
    ) \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      $(for python_version in $non_default_python_versions; do echo "python$python_version-dev python$python_version-distutils python$python_version-venv"; done) \
 && apt-get clean \
 && wget https://bootstrap.pypa.io/get-pip.py \
 && for python_version in $non_default_python_versions $default_python_version; do python$python_version get-pip.py || exit 1; done

# Boost
RUN set -x \
 && wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.gz \
 && tar xf boost_1_82_0.tar.gz \
 && cd boost_1_82_0 \
 && (for version in $LINCS_DEV_PYTHON_VERSIONS; do echo "using python : $version ;"; done) >tools/build/src/user-config.jam \
 && ./bootstrap.sh --with-libraries=python \
 && ./b2 $(for version in $LINCS_DEV_PYTHON_VERSIONS; do echo "python=$version"; done) \
 && cd /usr/local/include && ln -s /boost_1_82_0/boost boost \
 && cd /usr/local/lib && ln -s /boost_1_82_0/stage/lib/* .

# NVidia packages
RUN set -x \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
 && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
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
 && ldconfig

# Python packages: development and publication tools
RUN set -x \
 && pip3 install \
      auditwheel \
      semver \
      sphinx \
      sphinx-click \
      sphinx-jsonschema \
      sphinxcontrib-details-directive \
      twine \
 && for python_version in $LINCS_DEV_PYTHON_VERSIONS; do python$python_version -m pip install \
      build \
    || exit 1; done
# Recent patchelf version for 'auditwheel repair'
RUN set -x \
 && mkdir patchelf \
 && cd patchelf \
 && wget https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-x86_64.tar.gz \
 && tar xf patchelf-0.18.0-x86_64.tar.gz \
 && cp bin/patchelf /usr/local/bin

# Python packages: runtime dependencies
RUN for python_version in $LINCS_DEV_PYTHON_VERSIONS; do python$python_version -m pip install \
      click~=8.1 \
      matplotlib~=3.5 \
    || exit 1; done

# User
ARG UID
RUN useradd user --uid $UID --create-home
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user
USER user
RUN mkdir -p /home/user/.local/bin
ENV PATH=/home/user/.local/bin:$PATH:/usr/local/cuda-12.1/bin

# CCache
RUN ln -s $(which ccache) /home/user/.local/bin/x86_64-linux-gnu-gcc
RUN ln -s $(which ccache) /home/user/.local/bin/nvcc

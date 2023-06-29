# Copyright 2022 Vincent Jacques
# Copyright 2022 Laurent Cabaret

FROM ubuntu:20.04

# System packages
RUN set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      wget python3-pip `# Common tools` \
      software-properties-common dirmngr gpg-agent `# Required when adding the NVidia APT repository` \
      make cmake g++-8 `# Build tools` \
      libpng-dev `# Development version of direct dependencies` \
      valgrind `# Debug tools` \
 && apt-get clean

RUN set -x \
 && cd /usr/bin \
 && ln -s gcc-8 gcc \
 && ln -s g++-8 g++

# Google Test
RUN set -x \
 && wget https://github.com/google/googletest/archive/release-1.11.0.tar.gz \
 && tar xzf release-1.11.0.tar.gz \
 && cd googletest-release-1.11.0 \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make \
 && make install \
 && cd ../.. \
 && rm -r googletest-release-1.11.0 \
 && rm release-1.11.0.tar.gz

# NVidia packages
RUN set -x \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      cuda-cudart-dev-10-2 `# Development version of direct dependencies` \
      cuda-nvcc-10-2 `# Build tools` \
      cuda-memcheck-10-2 `# Debug tools` \
 && apt-get clean

# Put 'nvcc' in the PATH
ENV PATH="/usr/local/cuda-10.2/bin:${PATH}"

RUN set -x \
 && cd /usr/bin \
 && ln -sf gcc-8 gcc \
 && ln -sf g++-8 g++

# Python packages
RUN pip3 install \
      cpplint `# Static analysis tools`

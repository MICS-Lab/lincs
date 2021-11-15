# Copyright 2021 Vincent Jacques

FROM ubuntu:20.04

# System packages
RUN set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      wget python3-pip `# Common tools` \
      software-properties-common dirmngr gpg-agent `# Required when adding the NVidia APT repository` \
      make cmake g++ `# Build tools` \
&& apt-get clean

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
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
 && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      cuda-cudart-dev-11-2 `# Development version of direct dependencies` \
      cuda-nvcc-11-2 `# Build tools` \
 && apt-get clean

# Put 'nvcc' in the PATH
ENV PATH="/usr/local/cuda-11.2/bin:${PATH}"

# Python packages
RUN pip3 install \
      cpplint `# Static analysis tools`

# Copyright 2021-2022 Vincent Jacques

FROM ubuntu:20.04

# System packages
RUN set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      wget python3-pip graphviz `# Common tools` \
      software-properties-common dirmngr gpg-agent `# Required when adding the NVidia APT repository` \
      libboost-all-dev  `# Development version of direct dependencies` \
      make cmake g++ `# Build tools` \
      file gdb valgrind `# Debug tools` \
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
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      cuda-cudart-dev-11-2 libcurand-dev-11-2 `# Development version of direct dependencies` \
      cuda-nvcc-11-2 `# Build tools` \
 && apt-get clean

# Put 'nvcc' in the PATH
ENV PATH="/usr/local/cuda-11.2/bin:${PATH}"

# Python packages
RUN pip3 install \
      PyYaml  `# Common tools` \
      cpplint `# Static analysis tools`

# OR-Tools
# DO NOT UPGRADE to version 9.y: there is a SegFault bug (that I still need to report)
RUN set -x \
 && wget https://github.com/google/or-tools/releases/download/v8.2/or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && tar xf or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/include/* /usr/local/include \
 && cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/lib/*.so /usr/local/lib \
 && rm -r or-tools_Ubuntu-20.04-64bit_v8.2.8710 or-tools_ubuntu-20.04_v8.2.8710.tar.gz

# CLI11
RUN set -x \
 && cd /usr/local/include \
 && wget https://github.com/CLIUtils/CLI11/releases/download/v2.1.2/CLI11.hpp

# MagicEnum
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/Neargye/magic_enum/v0.7.3/include/magic_enum.hpp

# Lov-e-cuda
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/jacquev6/lov-e-cuda/fe6ea85/lov-e.hpp

# Chrones
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/LaurentCabaret/chrones/02c40fe0/c%2B%2B/chrones.hpp \
 && cd /usr/local/bin \
 && wget https://raw.githubusercontent.com/LaurentCabaret/chrones/02c40fe0/chrones-report.py \
 && chmod +x chrones-report.py

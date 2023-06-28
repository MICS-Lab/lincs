# Copyright 2023 Vincent Jacques

FROM ubuntu:22.04
# @todo Consider downgrading to ubuntu:20.04
#  - to support Python 3.5 and 3.6 (See "Supported Ubuntu and Python Versions" in https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)
#  - to provide slightly more usable manylinux_2_y wheels (i.e. with a smaller "y")

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
      python3-pip \
      python3-venv \
      software-properties-common \
      ssh \
      sudo \
      wget \
 && apt-get clean

# Python versions
ENV LINCS_DEV_PYTHON_VERSIONS="3.7 3.8 3.9 3.10 3.11"
RUN set -x \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      $( \
        for version in $LINCS_DEV_PYTHON_VERSIONS; do echo "python$version python$version-dev python$version-distutils python$version-venv"; done \
        | grep -v 3.10 `# Default version provided by Ubuntu` \
      ) \
 && apt-get clean

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

# Alglib
RUN set -x \
 && wget https://www.alglib.net/translator/re/alglib-4.00.0.cpp.gpl.tgz \
 && tar xf alglib-4.00.0.cpp.gpl.tgz \
 && cd alglib-cpp/src \
 && for f in *.cpp; do g++ -c -O3 -fPIC $f -o ${f%.cpp}.o; done \
 && ar rcs libalglib.a *.o \
 && mkdir /usr/local/include/alglib \
 && cp -r *.h /usr/local/include/alglib \
 && cp -r libalglib.a /usr/local/lib \
 && cd ../.. \
 && rm -r alglib-cpp alglib-4.00.0.cpp.gpl.tgz

# Valijson
RUN set -x \
 && wget https://github.com/tristanpenman/valijson/archive/refs/tags/v1.0.1.tar.gz \
 && tar xf v1.0.1.tar.gz \
 && cd valijson-1.0.1 \
 && ./bundle.sh yaml_cpp >/usr/local/include/valijson_yamlcpp_bundled.hpp

# Minisat
RUN set -x \
 && wget https://github.com/niklasso/minisat/archive/37dc6c67e2af26379d88ce349eb9c4c6160e8543.tar.gz \
 && tar xzf 37dc6c67e2af26379d88ce349eb9c4c6160e8543.tar.gz \
 && cd minisat-37dc6c67e2af26379d88ce349eb9c4c6160e8543 \
 `# Patch to compile with modern versions of g++` \
 && sed -i minisat/core/SolverTypes.h \
      -e 's/friend Lit mkLit(Var var, bool sign = false);/friend Lit mkLit(Var var, bool sign);/' \
      -e 's/inline  Lit  mkLit     (Var var, bool sign)/inline  Lit  mkLit     (Var var, bool sign = false)/' \
 && make config prefix=/usr/local \
 && make install \
 && ldconfig

# Python packages
RUN pip3 install \
      build \
      semver \
      sphinx \
      sphinx-click \
      sphinx-jsonschema \
      sphinxcontrib-details-directive \
      twine

# Python packages again, to speed-up development cycle
RUN for version in $LINCS_DEV_PYTHON_VERSIONS; \
    do \
      python$version -m pip install \
        click~=8.1 \
        matplotlib~=3.5 \
      || exit 1 \
    ; \
    done

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

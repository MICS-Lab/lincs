# Copyright 2023 Vincent Jacques

FROM ubuntu:22.04

RUN set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes \
      dirmngr \
      gpg-agent \
      libboost-python-dev \
      libyaml-cpp-dev \
      python3-dev \
      python3-pip \
      software-properties-common \
      wget \
 && apt-get clean

RUN set -x \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
 && add-apt-repository 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes \
      cuda-cudart-dev-12-1 \
      cuda-nvcc-12-1 \
 && apt-get clean

RUN set -x \
 && wget https://github.com/google/or-tools/releases/download/v8.2/or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && tar xf or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/include/* /usr/local/include \
 && cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/lib/*.so /usr/local/lib \
 && rm -r or-tools_Ubuntu-20.04-64bit_v8.2.8710 or-tools_ubuntu-20.04_v8.2.8710.tar.gz

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

RUN set -x \
 && wget https://github.com/tristanpenman/valijson/archive/refs/tags/v1.0.1.tar.gz \
 && tar xf v1.0.1.tar.gz \
 && cd valijson-1.0.1 \
 && ./bundle.sh yaml_cpp >/usr/local/include/valijson_yamlcpp_bundled.hpp

RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/Neargye/magic_enum/v0.8.2/include/magic_enum.hpp \
 && wget https://raw.githubusercontent.com/d99kris/rapidcsv/v8.75/src/rapidcsv.h \
 && wget https://raw.githubusercontent.com/jacquev6/lov-e-cuda/13e45bc/lov-e.hpp \
 && wget https://raw.githubusercontent.com/doctest/doctest/v2.4.11/doctest/doctest.h

RUN set -x \
 && wget https://github.com/niklasso/minisat/archive/37dc6c67e2af26379d88ce349eb9c4c6160e8543.tar.gz \
 && tar xzf 37dc6c67e2af26379d88ce349eb9c4c6160e8543.tar.gz \
 && cd minisat-37dc6c67e2af26379d88ce349eb9c4c6160e8543 \
 && sed -i minisat/core/SolverTypes.h \
      -e 's/friend Lit mkLit(Var var, bool sign = false);/friend Lit mkLit(Var var, bool sign);/' \
      -e 's/inline  Lit  mkLit     (Var var, bool sign)/inline  Lit  mkLit     (Var var, bool sign = false)/' \
 && make config prefix=/usr/local \
 && make install

ENV PATH=/usr/local/cuda-12.1/bin:$PATH
RUN ldconfig

ARG LINCS_VERSION
RUN pip3 install lincs==$LINCS_VERSION

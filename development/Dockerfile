FROM ubuntu:22.04

# System packages
RUN set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      ca-certificates \
      g++ \
      libboost-python-dev \
      libyaml-cpp-dev \
      python3-dev \
      python3-pip \
      wget \
 && apt-get clean

# MagicEnum
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/Neargye/magic_enum/v0.8.2/include/magic_enum.hpp

# RapidCSV
RUN set -x \
 && cd /usr/local/include \
 && wget https://raw.githubusercontent.com/d99kris/rapidcsv/v8.75/src/rapidcsv.h

# User
ARG UID
RUN useradd user --uid $UID --create-home
USER user
ENV PATH=/home/user/.local/bin:$PATH

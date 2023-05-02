FROM ubuntu:22.04

# System packages
RUN set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      g++ \
      libboost-python-dev \
      libyaml-cpp-dev \
      python3-dev \
      python3-pip \
 && apt-get clean

# User
ARG UID
RUN useradd user --uid $UID --create-home
USER user
ENV PATH=/home/user/.local/bin:$PATH

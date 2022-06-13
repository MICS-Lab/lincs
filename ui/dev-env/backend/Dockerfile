FROM nvidia/cuda:11.2.0-base-ubuntu20.04

RUN set -x \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      wget python3 python3-pip libboost-thread1.71.0 libgomp1 \
 && apt-get clean

# OR-Tools
RUN set -x \
 && wget https://github.com/google/or-tools/releases/download/v8.2/or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && tar xf or-tools_ubuntu-20.04_v8.2.8710.tar.gz \
 && cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/include/* /usr/local/include \
 && cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/lib/*.so /usr/local/lib \
 && rm -r or-tools_Ubuntu-20.04-64bit_v8.2.8710 or-tools_ubuntu-20.04_v8.2.8710.tar.gz

WORKDIR /application
COPY ui/backend/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

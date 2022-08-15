FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

# python install
ENV PYTHON_VERSION=3.8.10
ENV PYTHON_DOWNLOAD_URL https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz

ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install -y tcl


RUN apt-get -y install wget
RUN apt-get install -y --fix-missing \
        build-essential \
        cmake \
        wget \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
        liblzma-dev

RUN set -eux; \
    wget "$PYTHON_DOWNLOAD_URL"; \
    tar -xf Python-${PYTHON_VERSION}.tgz
RUN cd Python-${PYTHON_VERSION}; \
    ./configure  --enable-optimizations ; \
    make -j 8 ; \
    make install -j 8

#RUN rm Python-${PYTHON_VERSION}.tgz

ADD . /root
WORKDIR /root

COPY requirements.txt ./
RUN cat requirements.txt  | xargs -n 1 -L 1 pip3 install --no-cache-dir
RUN pip3 install -rrequirements.txt

COPY . .

ENTRYPOINT ["/bin/bash"]

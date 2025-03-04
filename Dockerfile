# syntax=docker/dockerfile:1
FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3\
    python3-pip\
    python3-venv\
    python3-dev\
    make\
    g++\
    build-essential\
    redis\
    curl &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /rad
COPY . /rad
RUN pip3 install --no-cache-dir .


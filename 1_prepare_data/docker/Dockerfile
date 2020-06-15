FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        zip \
        unzip \
        git \
        ca-certificates \
        curl \
        python3.6 \
        python3-pip \
        && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt --no-cache --upgrade

COPY code /opt/program

ENTRYPOINT ["python3", "/opt/program/prepare_data.py"]
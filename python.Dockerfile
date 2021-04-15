FROM python:3.8.9-slim-buster
# This Dockerfile is problematic because atari fails to install
RUN python -m pip install --upgrade pip

RUN apt-get update && apt-get -y install make  # for ataripy

WORKDIR home/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

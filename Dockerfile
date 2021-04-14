FROM python:3.8.9-slim-buster

RUN python -m pip install --upgrade pip
RUN apt-get -y install make  # for ataripy

WORKDIR home/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


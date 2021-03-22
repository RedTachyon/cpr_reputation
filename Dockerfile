FROM python:3.8.8-slim-buster

RUN python -m pip install --upgrade pip

WORKDIR home/
COPY . .

RUN pip install -r requirements.txt

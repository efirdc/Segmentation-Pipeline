FROM python:3.7.10-slim-buster

WORKDIR /Segmentation-Pipeline

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p /Segmentation-Pipeline/data/input/raw_data
RUN mkdir -p /Segmentation-Pipeline/data/input/preprocessed
RUN mkdir -p /Segmentation-Pipeline/data/output

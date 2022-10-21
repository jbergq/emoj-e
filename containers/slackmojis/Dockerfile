FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt update && apt-get update
RUN apt install npm -y

WORKDIR /workspace
COPY ext/slackmojis-dl .

CMD npx slackmojis-dl --dest datasets/slackmojis/temp

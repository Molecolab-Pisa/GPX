FROM opensuse/leap:latest
LABEL version="1.5.4"
LABEL description="Dockerfile to install and test GPX on gitlab"
WORKDIR /usr/src/app
RUN zypper --non-interactive install python310 python310-pip
RUN pip3 install --upgrade pip
RUN pip3 install virtualenv; virtualenv venv
COPY . .

# Build a docker image that can be used for inference
# Python3 image using nginx, gunicorn, flask

# LTS Support
FROM ubuntu:20.04

MAINTAINER Vignesh Selvaraj <vselvaraj@wisc.edu>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
      wget \
      python3-pip \
      python3-setuptools \
      nginx \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Force the symbolic link
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Install the python pakages required
RUN pip --no-cache-dir install numpy==1.19.5 scipy==1.5.3 scikit-learn==0.24.2 PyWavelets pandas flask gunicorn

# Setting some environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Copy the trained models to a specific location
COPY trained_models /opt/ml/models
# Copy the preprocessor - standardization
COPY preprocessors /opt/ml/preprocessors

# Setup the program in image
COPY models /opt/program
WORKDIR /opt/program

# Set the entry point
ENTRYPOINT ["python", "serve"]
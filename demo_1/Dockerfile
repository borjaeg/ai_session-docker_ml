# Base image
FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /tf

# Set desired Python version
ENV python_version 3.6

# Install desired Python version
RUN apt install -y python${python_version}

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN python -m pip install --upgrade pip setuptools wheel

# By copying over requirements first, we make sure that Docker will "cache"
# our installed requirements avoiding further reinstallations
COPY requirements.txt requirements.txt

# Install the requirements
RUN python -m pip install -r requirements.txt --no-cache-dir

# Port used by Jupyter
EXPOSE 8888

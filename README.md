# AI SESSION: From local notebooks to MLOps
## Initial set-up
#### Install Docker
```shell
sudo apt update
sudo apt upgrade # Warning: Only if you want to upgrade all the packages
sudo apt install docker.io
sudo docker --version
sudo systemctl status docker
sudo apt install docker-compose

sudo usermod -aG sudo nameOfUser
sudo usermod -aG docker nameOfUser
```

#### Create SSH Tunnel for Jupyter Notebooks
```shell
ssh -N -f -L localhost:8888:localhost:8888 user@server_ip # SSH Tunnel creation
ssh user@server_ip # Connect
``` 

## Basic Commands
#### Images
```shell
docker image ls  # List available images
docker image rm [image_id]  # Remove specific image
docker rmi --force [image_id]  # Remove specific image (even with other images/containers affected)
docker images -f dangling=true  # Remove dangling images
docker rmi $(docker images -a -q) # Remove all images
```  

#### Cotainers
```shell
docker container ls -a  # List containers
docker container ls –aq  # List containers' identifiers
docker container stop [container_id]  # Stop a container
docker container stop $(docker container ls –aq)  # Stop all containers
docker container rm [container_id]  # Remove a container
docker container prune  # Remove unused containers
docker rm $(docker ps -a -f status=exited -q)  # Remove containers with status = exit
docker ps -a -f status=exited -f status=created  # Remove containers with status = exit and created
docker ps -a | grep "pattern" | awk '{print $1}' | xargs docker rm  # Remove containers according to a pattern
```

#### Services
```shell
docker service ls # List services
```

## Tutorial 1
#### Install NVIDIA Support
```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)  
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -  
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit  # Install NVIDIA Support
sudo systemctl restart docker # Restart Docker
```

#### Run Nvidia-SMI
```shell
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi # Download CUDA image and create a container from it
```

#### Run Python VM and Tensorflow with GPU
```shell
docker run -it --rm --gpus all tensorflow/tensorflow:latest-gpu python # Download Tensorflow image and create a container from it
```

```python
# Check whether TF finds the GPU 
import tensorflow as tf  
tf.version.VERSION  
from tensorflow.python.client import device_lib  
print(device_lib.list_local_devices())
```
#### Run Jupyter notebook with GPU Tensorflow Examples
```shell
docker run -it --rm --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter # Download Tensorflow-Jupyter image and create a container from it.
```

#### Extend previous Image with this Dockerfile

```yaml
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
```

```shell
cd demo_1
docker build -t eden_library/laboratory:latest-gpu-jupyter .
docker run -it --rm --gpus all -p 8888:8888 eden_library/laboratory:latest-gpu-jupyter
```

**requirements.txt**
```text
numpy==1.19.2
matplotlib==2.2.2
pandas==1.1.5
scikit-learn==0.24.1
tensorflow_datasets==4.2.0
```

## Tutorial 2
```shell
cd demo_2
docker-compose up --build eden_lab
```
#### Training and export a deep model (localhost:8888)
<img src="https://user-images.githubusercontent.com/2207826/112990271-28ac3c00-9166-11eb-9591-7a882378e497.png" width="400px"></br>
<img src="https://user-images.githubusercontent.com/2207826/112990330-382b8500-9166-11eb-8e4b-9622b5740b89.png" width="400px">

#### Launch cluster and deploy trained model in the previous step
```shell
docker-compose build weed_classifier_service  # Build service and deploy previously trained model
docker swarm init  # Start cluster
docker stack deploy -c docker-compose.yml tf # Create workers (tf is the name if the stack, it can be changed)

docker ps # Check the workers are running
docker service ls # Check the workers are running

docker logs # Check the port where the client simulator is listening
```

#### Shutdown cluster and workers
```shell
docker stack rm tf # Create workers (tf is the name if the stack)
docker swarm leave --force
```
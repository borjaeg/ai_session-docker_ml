## Initial set-up
#### Create SSH Tunnel for Jupyter Notebooks
ssh -N -f -L localhost:8888:localhost:8888 user@server_ip
ssh user@server_ip

## Basic Commands
#### Images
```shell
docker image ls  
docker image rm [image_id]  
docker rmi --force [image_id]  
docker images -f dangling=true  
docker rmi $(docker images -a -q)
```  

#### Cotainers
```shell
docker container ls -a  
docker container ls –aq  
docker container stop [container_id]  
docker container stop $(docker container ls –aq)  
docker container rm [container_id]  
docker container prune  
docker rm $(docker ps -a -f status=exited -q)  
docker ps -a -f status=exited -f status=created  
docker ps -a | grep "pattern" | awk '{print $1}' | xargs docker rm  
```

#### Services
```shell
docker service ls
```

## Tutorial 1
#### Install NVIDIA Support
```shell
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)  
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -  
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit  
$ sudo systemctl restart docker
```

#### Run Nvidia-SMI
```shell
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
#### Run Python VM
docker run -it --rm --gpus all tensorflow/tensorflow:latest-gpu python
```

```python
import tensorflow as tf  
tf.version.VERSION  
from tensorflow.python.client import device_lib  
print(device_lib.list_local_devices())
```

#### Run Jupyter notebook with Tensorflow Example
docker run -it --rm --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
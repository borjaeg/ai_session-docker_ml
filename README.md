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
```

#### Run Python VM and Tensorflow with GPU
```shell
docker run -it --rm --gpus all tensorflow/tensorflow:latest-gpu python
```

```python
import tensorflow as tf  
tf.version.VERSION  
from tensorflow.python.client import device_lib  
print(device_lib.list_local_devices())
```
#### Run Jupyter notebook with GPU Tensorflow Examples
```shell
docker run -it --rm --gpus all -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

#### Extend previous Image
```shell
cd demo_1
docker build -t eden_library/laboratory:latest-gpu-jupyter .
docker run -it --rm --gpus all -p 8888:8888 eden_library/laboratory:latest-gpu-jupyter
```
requirements.txt
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

```shell
docker-compose build weed_classifier_service
docker swarm init
docker stack deploy -c docker-compose.yml tf

docker ps
docker service ls

docker logs
docker stack rm tf
docker swarm leave --force
```

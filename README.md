# AI SESSION: From local notebooks to MLOps with Docker
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
docker build -t eden_library/laboratory:latest-gpu-jupyter . # Build the image 
docker run -it --rm --gpus all -p 8888:8888 eden_library/laboratory:latest-gpu-jupyter # Run the container 
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

```yaml
version: "3.7"

services:

  eden_lab:
    image: jupyter/tensorflow-notebook
    ports:
      - "8888:8888"
    volumes:
      - "./notebooks:/home/jovyan/projectDir"
    environment:
      - "JUPYTER_ENABLE_LAB=yes"

  weed_classifier_service:
    image: eden_library:cotton_model
    build:
      context: .
      dockerfile: Dockerfile
    deploy:
      replicas: 4
      endpoint_mode: vip
```

```shell
cd demo_2
docker-compose up --build eden_lab # Build the image and run as a container. A Jupyter Lab will be launched.
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
# AI Session: TensorBoard
Presentation available at:
- https://docs.google.com/presentation/d/1z01UvzbP9RDy1dtXs5AMRB0_eWZiMW91dFDpmmur4WE/edit?usp=sharing


--- 
## Demo 1
--- 
### Docker cleanup
```
sudo docker container ls -a
sudo docker container stop XXX
sudo docker system prune
```

### Run Docker container with TF-GPU, Jupyter, TensorBoard:
`sudo docker run --runtime=nvidia -it --rm -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:2.4.1-gpu-jupyter`

### Load the TensorBoard notebook extension 
```
%load_ext tensorboard
%tensorboard --logdir logs --bind_all
```
```
from tensorboard import notebook
notebook.list() # View open TensorBoard instances
notebook.display(port=6006, height=1000)
```

### Check GPU availability 
```
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

### Run TF without GPU on simple scalar task
```
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

import numpy as np
data_size = 1000
train_pct = 0.8

train_size = int(data_size * train_pct)

x = np.linspace(-1, 1, data_size)
np.random.shuffle(x)
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size, ))

x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

logdir = "logs/no_gpu/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=1),
    keras.layers.Dense(1),
])

model.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(lr=0.2),
)

print("Training ... With default parameters, this takes less than 10 seconds.")
training_history = model.fit(
    x_train, # input
    y_train, # output
    batch_size=train_size,
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=40,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)

print("Average test loss: ", np.average(training_history.history['loss']))

print(model.predict([60, 25, 2]))
```

### Run TF with GPU on MNIST
```
import tensorflow as tf
import datetime, os

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

def train_model():

    model = create_model()
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    logdir = os.path.join("logs/gpu", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model.fit(x=x_train, 
            y=y_train, 
            epochs=20, 
            validation_data=(x_test, y_test), 
            callbacks=[tensorboard_callback])

train_model()
```


--- 
## Demo 2
--- 
### Initiate TensorBoard
A level above `logs` folder:
`tensorboard --logdir=logs`
Open `http://localhost:6006/` in browser


### Run PyTorch script
In folder of `train_tomato.py`:
```
conda activate eden_pytorch
python train.py
```

### Check GPU performance
`nvidia-smi`


---
## Links
---
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian
- https://www.tensorflow.org/install/docker#gpu_support
- https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks
- https://www.tensorflow.org/tensorboard/scalars_and_keras
- https://www.machinecurve.com/index.php/2019/11/13/how-to-use-tensorboard-with-keras/
- https://pytorch.org/docs/stable/tensorboard.html

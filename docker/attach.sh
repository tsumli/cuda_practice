#!/bin/bash
IMAGE_NAME="cuda_practice:0.1.0"
CONTAINER_NAME="cuda_practice_container"

function check_exists {
    if [ -z "$(docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
        return 1
    fi
    return 0
}

check_exists
if [ $? -eq 0 ]; then
    echo "Container $CONTAINER_NAME exists. Attaching..."
    docker start $CONTAINER_NAME
    docker attach $CONTAINER_NAME
else
    echo "Container $CONTAINER_NAME does not exist. Creating a new one..."
    docker run -it \
        --name $CONTAINER_NAME \
        --gpus all \
        -v $PWD:/workspace \
        -v /usr/local/NVIDIA-Nsight-Compute:/usr/local/NVIDIA-Nsight-Compute \
        -v /opt/nvidia/nsight-systems/2024.5.1:/opt/nvidia/nsight-systems/2024.5.1 \
        cuda_practice:0.1.0 /bin/bash
fi

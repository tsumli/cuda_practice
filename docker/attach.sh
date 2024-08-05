#!/bin/bash
docker run -it --rm --gpus all \
    -v $PWD:/workspace \
    -v /usr/local/NVIDIA-Nsight-Compute:/usr/local/NVIDIA-Nsight-Compute \
    -v /opt/nvidia/nsight-systems/2024.5.1:/opt/nvidia/nsight-systems/2024.5.1 \
    cuda_practice:0.1.0 /bin/bash

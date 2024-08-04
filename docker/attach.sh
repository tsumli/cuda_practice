#!/bin/bash
docker run -it --rm --gpus all -v $PWD:/workspace cuda_practice:0.1.0 /bin/bash

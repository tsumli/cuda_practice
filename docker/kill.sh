#!/bin/bash
CONTAINER_NAME="cuda_practice_container"

docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

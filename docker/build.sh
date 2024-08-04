#!/bin/bash
NAME=$(basename $(git rev-parse --show-toplevel))
VERSION="0.1.0"

echo "Building $NAME:$VERSION"
docker build -t $NAME:$VERSION --file docker/Dockerfile --network host . 
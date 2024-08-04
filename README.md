# Cuda Practice

## Build docker image

By default, `cuda_practice:0.1.0` will be built.

```bash
bash docker/build.sh
```

## Compile

```bash
bash docker/attach.sh

# inside docker
mkdir build && cd docker
cmake ..
```

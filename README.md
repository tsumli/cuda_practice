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
mkdir build && cd build
cmake .. -G "Ninja"
ninja
```

I checked codes can be compiled with `NVIDIA GeForce RTX 4060 Laptop GPU`.
<details>
<summary>Detail of device information used on experiments</summary>
  
```text
Device Number: 0
  Device name: NVIDIA GeForce RTX 4060 Laptop GPU
  Memory Clock Rate (KHz):8001000
  Warp size: 32
  Max threads per block: 1024
  Max threads per multiprocessor: 1536
  Max threads per block dimension: (1024, 1024, 64)
  Max grid size: (2147483647, 65535, 65535)
  Total global memory (bytes): 8325824512
  Shared memory per block (bytes): 49152
  Memory bus width (bits): 128
  Compute capability: 8.9
```
</details>

## Acknowledgement

- https://github.com/NVIDIA/cuda-samples

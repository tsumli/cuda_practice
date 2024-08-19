/**
 * @file main.cu
 * @ref
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "common/cuda/exception.h"

int main() {
    cudaGraph_t graph;
    cudaStream_t stream;
    THROW_IF_FAILED(cudaStreamCreate(&stream));
    THROW_IF_FAILED(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Create the graph - it starts out empty
    THROW_IF_FAILED(cudaGraphCreate(&graph, 0));

    // For the purpose of this example, we'll create
    // the nodes separately from the dependencies to
    // demonstrate that it can be done in two stages.
    // Note that dependencies can also be specified
    // at node creation.
    cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
    cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
    cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
    cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

    // Now set up dependencies on each node
    cudaGraphAddDependencies(graph, &a, &b, 1);  // A->B
    cudaGraphAddDependencies(graph, &a, &c, 1);  // A->C
    cudaGraphAddDependencies(graph, &b, &d, 1);  // B->D
    cudaGraphAddDependencies(graph, &c, &d, 1);  // C->D
}
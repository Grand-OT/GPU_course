#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#define BASE_TYPE double
#define M_PI  3.141592653
#define n 1e8

__global__ void sinMass(BASE_TYPE* A, int arraySize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < arraySize)
        A[index] = sin((BASE_TYPE)((index % 360) * M_PI / 180));
}

__global__ void sinfMass(BASE_TYPE* A, int arraySize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < arraySize)
        A[index] = sinf((BASE_TYPE)((index % 360) * M_PI / 180));
}

__global__ void sinCudaMass(BASE_TYPE* A, int arraySize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < arraySize)
        A[index] = __sinf((BASE_TYPE)((index % 360) * M_PI / 180));
}

double calcSinError(void(*fn)(BASE_TYPE*, int), BASE_TYPE* arr_cpu, BASE_TYPE* arr_gpu,
    unsigned gridSize, unsigned blockSize);

int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0)
    {
        std::cout << "No CUDA devices detected" << std::endl;
        return 0;
    }
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    int blockSize = devProp.maxThreadsPerBlock;
    
    unsigned gridSize = (n + blockSize - 1) / blockSize;

    BASE_TYPE* arr_gpu;
    BASE_TYPE* arr_cpu = new BASE_TYPE[n];

    cudaMalloc(&arr_gpu, n * sizeof(BASE_TYPE));

    
    double sinErr = calcSinError(sinMass, arr_cpu, arr_gpu, gridSize, blockSize);
    std::cout << "Sin error: " << sinErr << std::endl;

    double sinfErr = calcSinError(sinfMass, arr_cpu, arr_gpu, gridSize, blockSize);
    std::cout << "Sinf error: " << sinfErr << std::endl;

    double sinfCudaErr = calcSinError(sinCudaMass, arr_cpu, arr_gpu, gridSize, blockSize);
    std::cout << "Sinf CUDA error: " << sinfCudaErr << std::endl;

    delete[] arr_cpu;
    cudaFree(arr_gpu);

    return 0;
}

double calcSinError(void(*fn)(BASE_TYPE*, int), BASE_TYPE* arr_cpu, BASE_TYPE* arr_gpu,
    unsigned gridSize, unsigned blockSize)
{
    (*fn) << < gridSize, blockSize >> > (arr_gpu, n);

    cudaMemcpy(arr_cpu, arr_gpu, n * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
    double err = 0;
    for (int i = 0; i < n; i++) {
        err += fabs(sin((i % 360) * M_PI / 180.0) - arr_cpu[i]);
    }
    return err /= n;

}
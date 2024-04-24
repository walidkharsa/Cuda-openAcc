#include <stdio.h>
#include <cuda.h>

#define N 2048 // Square matrix

__global__ void matrixMul(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    int *a, *b, *c;            
    int *d_a, *d_b, *d_c;      
    int size = N * N * sizeof(int);
    float time;
    cudaEvent_t start, stop;

    
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    
    a = (int *)malloc(size); 
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the kernel: %f ms\n", time);

    
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

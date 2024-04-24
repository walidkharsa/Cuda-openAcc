#include <stdio.h>
#include <cuda.h>

#define N 2048  // Square matrix
#define TILE_WIDTH 16  

__global__ void tiledMatrixMul(int *a, int *b, int *c) {
    __shared__ int tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tile_b[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int sum = 0;

    for (int m = 0; m < (N / TILE_WIDTH); ++m) {
        tile_a[threadIdx.y][threadIdx.x] = a[row * N + (m * TILE_WIDTH + threadIdx.x)];
        tile_b[threadIdx.y][threadIdx.x] = b[(m * TILE_WIDTH + threadIdx.y) * N + col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    c[row * N + col] = sum;
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

    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    tiledMatrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the tiled kernel: %f ms\n", time);

    
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

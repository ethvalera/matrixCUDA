//
//  matrixCUDA.cu
//  
//

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void matrixOperation(int *m, int *v, int *c, int N, int M){
    
    // Calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // AVG column
    if (row < M && col < N) {
        int temp = 0;
        for (int i = 0; i<M; i++){
            temp += m[i * N + col];
        }
        v[col] = temp/M;
    }
    
    // MOV matrix (rolling average)
    if (row < M && col < N) {
        for (int i = 0; i < N; i++){
            int count = 0;
            int tmp = 0;
            for (int j = i; (j>(i-9)) && (j>=0); j--){
                tmp += m[row * N + j];
                count++;
            }
            c[row * N + i] = tmp/count;
            }
        }
}


int main (){
    // Set matrix dimensions
    const int N = 1000;
    const int M = 10;
    const int msize = N * M * sizeof(int);
    const int vsize = N * sizeof(int);
    
    // Allocate memory for matrices
    int *DATA, *AVG, *MOV, *ad, *bd, *cd;
    cudaMalloc((void**)&ad, msize);
    cudaMalloc((void**)&bd, vsize);
    cudaMalloc((void**)&cd, msize);
    cudaMalloc((void**)&DATA, msize);
    cudaMalloc((void**)&AVG, vsize);
    cudaMalloc((void**)&MOV, msize);
    
    // Init. DATA randomly
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            DATA[N*i+j] = rand() % 100;
        }
    }

    // Copy memory to GPU
    cudaMemcpy(ad, DATA, msize, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, AVG, vsize, cudaMemcpyHostToDevice);
    cudaMemcpy(cd, MOV, msize, cudaMemcpyHostToDevice);
    
    // Number of threads per block
    int THREADS_M = 16;
    
    // Number of blocks for matrix
    int blocks_rows = (M + THREADS_M - 1) / THREADS_M;
    int blocks_col = (N + THREADS_M - 1) / THREADS_M;
    
    // 2D blocks size for matrix
    dim3 dimBlock(THREADS_M, THREADS_M);
    
    // 2D grid for matrix
    dim3 dimGrid(blocks_col, blocks_rows);
    
    // Call the matrixOperation kernel
    matrixOperation<<<dimGrid, dimBlock>>>(ad, bd, cd, N, M);
    
    // Copy memory back to the CPU
    cudaMemcpy(AVG, bd, vsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(MOV, cd, msize, cudaMemcpyDeviceToHost);
    cudaFree(bd);
    cudaFree(cd);
    
    return EXIT_SUCCESS;
}


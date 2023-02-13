#include <cuda_runtime.h>
#include <iostream>

#define N 3

__global__ void matrixAddition(double* A, double* B, double* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

__global__ void matrixSubtraction(double* A, double* B, double* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] - B[i * N + j];
    }
}

__global__ void matrixDivision(double* A, double* B, double* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] / B[i * N + j];
    }
}

__global__ void matrixMultiplication(double* A, double* B, double* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        double sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
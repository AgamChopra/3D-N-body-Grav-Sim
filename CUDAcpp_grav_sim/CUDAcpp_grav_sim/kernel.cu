
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"

#include <stdio.h>

int main() {
    double A[N][N] = { {1.2, 9.2, 4.3}, {4.84, 0.25, 7.126}, {327.7, 568.1, 9.452} };
    double B[N][N] = { {10.5, 20.45534, 30.45}, {40.4566, 50.634, 60.45}, {70.563, 80.34, 90.54} };
    double C[N][N];

    double* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(double));
    cudaMalloc((void**)&d_B, N * N * sizeof(double));
    cudaMalloc((void**)&d_C, N * N * sizeof(double));

    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);

    // Matrix Addition
    matrixAddition << <gridSize, blockSize >> > (d_A, d_B, d_C);
    cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Matrix Addition: " << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Matrix Subtraction
    matrixSubtraction << <gridSize, blockSize >> > (d_A, d_B, d_C);
    cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Matrix Subtraction: " << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Matrix Division
    matrixDivision << <gridSize, blockSize >> > (d_A, d_B, d_C);
    cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Matrix Division: " << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Matrix Multiplication
    matrixMultiplication << <gridSize, blockSize >> > (d_A, d_B, d_C);
    cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Matrix Multiplication: " << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
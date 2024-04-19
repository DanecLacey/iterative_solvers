#include <stdio.h>
#include <cstdlib>
#include <algorithm>

#ifndef __CUDACC__
    #include <omp.h>
#endif

#ifdef __CUDACC__
__global__
void saxpy_gpu(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a*x[i] + y[i];
}
#endif

void saxpy_cpu(int n, float a, float *x, float *y)
{
    #pragma omp parallel for
    for(int i = 0; i < n; ++i){
        y[i] = a*x[i] + y[i];
    }
}

int main(void) {
    int N = 1 << 30;

    // Declare host and device pointers separately
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float)); // How to best do with vectors?
    y = (float*)malloc(N*sizeof(float));

    // Populate host arrays
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

#ifdef __CUDACC__
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    // Literally in the name
    // Signature: (dest, src, cnt, kind)
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
#endif

#ifdef __CUDACC__
    // Launch kernel
    saxpy_gpu<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y); // How to choose?
#else
    saxpy_cpu(N, 2.0f, x, y);
#endif

#ifdef __CUDACC__
    // Copy result back to host
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost); //<- again, look at the name
#endif
    // Assess error
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i)
        maxError = std::max(maxError, std::abs(y[i]-4.0f));
    printf("Max error: %f\n", maxError);

    // Free memory
#ifdef __CUDACC__
    cudaFree(d_x);
    cudaFree(d_y);
#endif
    free(x);
    free(y);
}

#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add(int n, double *x, double *y)
{ //One block mode
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x; //Get number of threads in block
  for (int i = index; i < n; i+=stride)
    y[i] = sin(x[i]*x[i]) + y[i];
}

int main(void)
{
  int N = 1<<25;
  double *x, *y;

  // Allocate Unified Memory accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(double));
  cudaMallocManaged(&y, N*sizeof(double));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<100, 256>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  double maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}

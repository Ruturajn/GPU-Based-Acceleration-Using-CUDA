#include <iostream>
#include <iomanip>
#include <assert.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;

#define THREADSx 32
#define THREADSy 32

/****************************** Per_Row Add Kernel ***************************/
__global__ void per_row_kernel(int m, int n, int *A, int *B, int *C)
{
    int mat_ind = (threadIdx.x + (blockDim.x * blockIdx.x));

    if (mat_ind < m)
    {
        int ind = (mat_ind * n);
        for (int i = 0; i < n; i++)
        {
            C[ind+i] = A[ind+i] + B[ind+i];
        }
    }
}

/************************** Per_Column Add Kernel ****************************/
__global__ void per_column_kernel(int m, int n, int *A, int *B, int *C)
{
    int col = (threadIdx.y + (blockDim.y * blockIdx.x));

    int ind = 0;
    if (col < n)
    {
        for (int i = 0; i < m; i++)
        {
            ind = (i * n + col);
            C[ind] = A[ind] + B[ind];
        }
    }
}

/************************* Per_Element Add Kernel ****************************/
__global__ void per_element_kernel(int m, int n, int *A, int *B, int *C)
{

    int row = (threadIdx.y + (blockDim.y * blockIdx.y));
    int col = (threadIdx.x + (blockDim.x * blockIdx.x));

    if (row < m && col < n)
    {
        int ind = (row * n + col);
        C[ind] = A[ind] + B[ind];
    }
}

/************************ Reference CPU Add Function *************************/
void refAdd(int *h_a, int *h_b, int *h_c, int m, int n, int *h_gpu_res)
{
    for (int i = 0; i < (m); i++)
    {
        for (int j = 0; j < (n); j++)
        {
            int ind = (i * n + j);
            h_c[ind] = h_a[ind] + h_b[ind];
            assert(h_gpu_res[ind] == h_c[ind]);
        }
    }

    cout << "" << endl;
    cout << "====================================" << endl;
    cout << "COMPUTATION VERIFIED SUCCESSFULLY !!" << endl;
    cout << "====================================" << endl;
}

int main(void)
{
    // Getting input from the user to determine which kernel should be run
    // and num_rows and num_cols
    int num_rows, num_cols;
    cout << "" << endl;
    cout << "Enter the number of rows and columns : ";
    cin >> num_rows >> num_cols;


    int argc;
    cout << "" << endl;
    cout << "Enter the kernel that should be run [(0 -> perRow) / (1 -> perColumn) / (2 -> perElement)] : ";
    cin >> argc;
    
    // Creating host variables
    int *h_a, *h_b, *h_c, *h_res_gpu;

    // Allocating memory for host variables
    size_t arr_size = (sizeof(int) * num_rows * num_cols);
    h_a = (int *)malloc(arr_size);
    h_b = (int *)malloc(arr_size);
    h_c = (int *)malloc(arr_size);
    h_res_gpu = (int *)malloc(arr_size);

    // Intializing the host arrays with random 2 digit integers
    for (int i = 0; i < (num_rows * num_cols); i++)
    {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Creating device variables
    int *A, *B, *C;

    // Allocating memory for the device variables
    cudaMalloc((void **)&A, arr_size);
    cudaMalloc((void **)&B, arr_size);
    cudaMalloc((void **)&C, arr_size);

    // Copying data from host to device
    cudaMemcpy(A, h_a, arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_b, arr_size, cudaMemcpyHostToDevice);

    int BLOCKSx = 0;
    int BLOCKSy = 0;

    // Create Events to time the kernels
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float exec_time;

    if (argc == 0)
    {
        cout << "-----------------------------" << endl;
        cout << "Running per_row_kernel ...." << endl;

        // Calculating the blocks required for kernel call
        BLOCKSy = (num_rows + THREADSy - 1) / THREADSy;

        dim3 threads(THREADSy);
        dim3 blocks(BLOCKSy);

        // Call the perElementKernel
        cudaEventRecord(start);
        per_row_kernel<<<blocks, threads>>>(num_rows, num_cols, A, B, C);
        cudaDeviceSynchronize();
    }
    else if (argc == 1)
    {
        cout << "-----------------------------" << endl;
        cout << "Running per_column_kernel ...." << endl;

        // Calculating the blocks required for kernel call
        BLOCKSx = (num_cols + THREADSx - 1) / (THREADSx);

        dim3 threads(1, THREADSx);
        dim3 blocks(BLOCKSx);

        // Call the perElementKernel
        cudaEventRecord(start);
        per_column_kernel<<<blocks, threads>>>(num_rows, num_cols, A, B, C);
        cudaDeviceSynchronize();
    }
    else if (argc == 2)
    {
        cout << "-----------------------------" << endl;
        cout << "Running per_element_kernel ...." << endl;

        // Calculating the blocks required for kernel call
        BLOCKSy = (num_rows + THREADSy - 1) / THREADSy;
        BLOCKSx = (num_cols + THREADSx - 1) / THREADSx;

        dim3 threads(THREADSx, THREADSy);
        dim3 blocks(BLOCKSx, BLOCKSy);

        // Call the perElementKernel
        cudaEventRecord(start);
        per_element_kernel<<<blocks, threads>>>(num_rows, num_cols, A, B, C);
        cudaDeviceSynchronize();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        free(h_a);
        free(h_b);
        free(h_c);
        exit(-1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exec_time, start, stop);

    // Copy the result back to host
    cudaMemcpy(h_res_gpu, C, arr_size, cudaMemcpyDeviceToHost);

    cout << "The time required to execute " << num_rows << " x " 
         << num_cols << " matrix addition by the selected kernel is " << exec_time << " milliseconds" << endl;

    // Verify the result
    auto start_cpu = std::chrono::high_resolution_clock::now();
    refAdd(h_a, h_b, h_c, num_rows, num_cols, h_res_gpu);
    auto diff = std::chrono::high_resolution_clock::now() - start_cpu;
    auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(diff);
    cout << "Performance of Thresholding on CPU - Time Taken : " << t1.count() << " microseconds" << endl;

    // Free all the allocated memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

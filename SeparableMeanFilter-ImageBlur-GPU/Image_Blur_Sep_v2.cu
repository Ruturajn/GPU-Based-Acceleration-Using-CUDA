#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;
using namespace cv;

#define kernel_width 9
#define THREADSx     8
#define THREADSy     8
#define BLOCK_PADDED (kernel_width + THREADSx)

/**************************Column Convolution Kernel******************************/
__global__ void ImgBlurColumnKernel(unsigned char* d_img, int num_rows, int num_cols,
                                    float* d_filter_c, unsigned char* d_temp_res){

    int col = (threadIdx.x + (blockDim.x*blockIdx.x));
    int row = (threadIdx.y + (blockDim.y*blockIdx.y));

    if (row < num_rows && col < num_cols){

        float prod_r = 0.0f, prod_g = 0.0f, prod_b = 0.0f;
        float img_val_r = 0, img_val_g = 0, img_val_b = 0;

        int temp_id_conv = 0;
        float filter_val = 0.0f;
        int row_val = 0;
        
        for (int i=0;i<kernel_width;i++){

            row_val = row + i - kernel_width/2;
            row_val = (row_val <= 0) ? 0 : ((row_val >= num_rows) ? (num_rows-1) : row_val);

            temp_id_conv = ((row_val*num_cols + col)*3);
            filter_val = d_filter_c[i];

            img_val_r = float(d_img[temp_id_conv]);
            img_val_g = float(d_img[temp_id_conv + 1]);
            img_val_b = float(d_img[temp_id_conv + 2]);

            prod_r += (img_val_r * filter_val);
            prod_g += (img_val_g * filter_val);
            prod_b += (img_val_b * filter_val);

            // if (blockIdx.x == 4 && blockIdx.y == 4 && threadIdx.x == 0 && threadIdx.y == 0)
            //     printf("The row and column for column is (%d,%d), for temp_id %d\n", row_val, col, temp_id_conv);
        }

        int idx = ((row*num_cols + col)*3);

        d_temp_res[idx] = prod_r;
        d_temp_res[idx + 1] = prod_g;
        d_temp_res[idx + 2] = prod_b;
    }
}


/**************************Row Convolution Kernel******************************/
__global__ void ImgBlurRowKernel(unsigned char* d_temp_res, int num_rows, 
                                 int num_cols,float* d_filter_r,
                                 unsigned char* d_res){
    
    int col = (threadIdx.x + (blockDim.x*blockIdx.x));
    int row = (threadIdx.y + (blockDim.y*blockIdx.y));
    if (row < num_rows && col < num_cols){
        
        float prod_r = 0.0f, prod_g = 0.0f, prod_b = 0.0f;
        float img_val_r = 0, img_val_g = 0, img_val_b = 0;
        
        int temp_id_conv = 0;
        float filter_val = 0.0f;
        int col_val = 0;

        for (int j=0;j<kernel_width;j++){

            col_val = col + j - kernel_width/2;
            col_val = (col_val <= 0) ? 0 : ((col_val >= num_cols) ? (num_cols-1) : col_val);

            temp_id_conv = ((row*num_cols + col_val)*3);
            filter_val = d_filter_r[j];

            img_val_r = float(d_temp_res[temp_id_conv]);
            img_val_g = float(d_temp_res[temp_id_conv + 1]);
            img_val_b = float(d_temp_res[temp_id_conv + 2]);

            prod_r += (img_val_r * filter_val);
            prod_g += (img_val_g * filter_val);
            prod_b += (img_val_b * filter_val);

            // if (blockIdx.x == 4 && blockIdx.y == 4) //&& threadIdx.x == 0 && threadIdx.y == 0)
            //     printf("The row and column for row is (%d,%d), for temp_id is %d\n", row, col_val, temp_id_conv);
        }

        int idx = ((row*num_cols + col)*3);
        
        d_res[idx] = prod_r;
        d_res[idx + 1] = prod_g;
        d_res[idx + 2] = prod_b;
    }
}


/*************************Pre-Processing Function*****************************/
void ImgBlur(Mat& h_img, Mat& h_res, float* filter_r, float* filter_c, 
             int filter_width){

        // Create events to time the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float exec_time;
        
        // Calculating the size of the frame for memory
        size_t img_size = (h_img.rows * h_img.cols * h_img.channels());

        // Calculating the size of the filter
        size_t filter_size = (filter_width * sizeof(float));

        // Defining Device variables
        unsigned char *d_frame, *d_temp_res,*d_res;
        float *d_filter_r, *d_filter_c;

        // Allocating memory
        cudaMalloc((void**)&d_frame, img_size);
        cudaMalloc((void**)&d_temp_res, img_size);
        cudaMalloc((void**)&d_res, img_size);
        cudaMalloc((void**)&d_filter_r, filter_size);
        cudaMalloc((void**)&d_filter_c, filter_size);

        // Copying frame to device
        cudaMemcpy(d_frame, h_img.ptr(), img_size, cudaMemcpyHostToDevice);

        // Copying filter from host to device
        cudaMemcpy(d_filter_r, filter_r, filter_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter_c, filter_c, filter_size, cudaMemcpyHostToDevice);

        // Configuring threads and blocks
        int BLOCKSx = (h_img.cols + THREADSx -1) / THREADSx;
        int BLOCKSy = (h_img.rows + THREADSy -1) / THREADSy;

        dim3 threads(THREADSx, THREADSy);
        dim3 blocks(BLOCKSx, BLOCKSy);

        // Launching the kernel
        cudaEventRecord(start);
        ImgBlurColumnKernel<<<blocks, threads>>>(d_frame, h_img.rows, h_img.cols,
                                           d_filter_r, d_temp_res);

        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess){
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        ImgBlurRowKernel<<<blocks, threads>>>(d_temp_res, h_img.rows, h_img.cols,
                                              d_filter_r, d_res);

        cudaDeviceSynchronize();

        err = cudaGetLastError();

        if (err != cudaSuccess){
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&exec_time, start, stop);

        // Copying the results back to host
        cudaMemcpy(h_res.ptr(), d_res, img_size, cudaMemcpyDeviceToHost);

        std::cout << "Time required to execute the kernel is : " << exec_time << endl;

        // Free the allocated memory
        cudaFree(d_frame);
        cudaFree(d_temp_res);
        cudaFree(d_res);
        cudaFree(d_filter_r);
        cudaFree(d_filter_c);

        return;
}


int main(void){

    // Reading the image

    Mat h_img = imread("../Images/index.jpeg", 1);
    //Mat h_img = imread("../Images/cinque_terre_small.jpg", 1);
    //Mat h_img = imread("/usr/share/backgrounds/brad-huchteman-stone-mountain.jpg", 1);
    //Mat h_img = imread("../Images/138728.jpg", 1);
    //Mat h_img = imread("../Images/2040735.jpg", 1);
    //Mat h_img = imread("../Images/UI-Sidewalk-640x480.jpg", 1);

    cout << h_img.cols << " || " << h_img.rows << endl;

    // Resizing the image
    // if (h_img.cols % 8 != 0 && h_img.rows % 8 != 0){
    //     resize(h_img, h_img, Size((h_img.cols/THREADSx + 1)*THREADSx, (h_img.rows/THREADSy + 1)*THREADSy));
    // }
    // cout << h_img.cols << " || " << h_img.rows << endl;

    // Create the output image
    Mat h_res(h_img.rows, h_img.cols, CV_8UC3);

    // Calculate the filter size and allocate memory for it
    //size_t filter_size = (kernel_width * sizeof(float));
    //float* h_filter_r = (float*)malloc(filter_size);
    //float* h_filter_c = (float*)malloc(filter_size);

    // Call the create_filter to fill the filter
    //create_filter(h_filter);
    float h_filter_r[9] = {1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f,
                           1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f};
    
    float h_filter_c[9] = {1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f,
                           1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f};


    // Call the pre-processing function, that further calls the kernel
    ImgBlur(h_img, h_res, h_filter_r, h_filter_c, kernel_width);

    // Calculate the difference based on the in-built function in OpenCV
    Mat ref_img; //= imread("../Images/Blurred_CPU.png", 1);
    //cv::GaussianBlur(h_img, ref_img, Size2i(kernel_width, kernel_width), 0, 0, BORDER_REPLICATE);
    boxFilter(h_img, ref_img, -1, Size2i(kernel_width, kernel_width), Point(-1,-1), true, BORDER_REPLICATE);
    Mat diff_opencv = abs(ref_img - h_res);

    // Show the results
    cv::imshow("Original", h_img);
    cv::imshow("Blurred", h_res);
    //cv::imshow("CV-Blur", ref_img);
    cv::imshow("diff-opencv", diff_opencv);

    // Saving Images to disk
    //imwrite("../Images/Blurred_Shared_mem_final_fruits.png", h_res);
    imwrite("../Images/Sep_Box_Filter.png", diff_opencv);
    //imwrite("../Images/Blurred_Sep.png",h_res);
    //imwrite("../Images/Reference_Blur_OpenCV.png", ref_img);

    // Wait for any keystrokes
    cv::waitKey(0);

    // Free the memory allocated to the filter
    //delete[] h_filter;

    return 0;
}

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
#define THREADSx     16
#define THREADSy     16

/*****************************Convolution Kernel******************************/
__global__ void ImgBlurKernel(unsigned char* d_img, int num_rows, int num_cols,
                              float* d_filter, int filter_width, 
                              unsigned char* d_res){
    
    // Calculating row and column indices
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);

    __shared__ unsigned char temp_img[THREADSx*THREADSy*3];
    
    // Making sure we don't access stuff out of the image
    if (row < num_rows && col < num_cols){
        
        // Loading Image data and filter into shared memory
        int temp_id = ((threadIdx.y*THREADSy + threadIdx.x)*3);
        int img_id = ((row*num_cols + col)*3);
        
        temp_img[temp_id] = d_img[img_id];
        temp_img[temp_id + 1] = d_img[img_id + 1];
        temp_img[temp_id + 2] = d_img[img_id + 2];

        // Barrier for all threads
        __syncthreads();

        // Initializing temporary product variable
        float prod_r = 0.0f;
        float prod_g = 0.0f;
        float prod_b = 0.0f;

        float img_val_r = 0;
        float img_val_g = 0;
        float img_val_b = 0;

        if ( (threadIdx.y < filter_width/2) || (threadIdx.x < filter_width/2) ||
                (threadIdx.y >= (THREADSx-filter_width/2)) || (threadIdx.x >= (THREADSy-filter_width/2))){


            // Defining starting points so that the filter doesn't fall off the image
            int start_row = row - (filter_width/2);
            int start_col = col - (filter_width/2);

            for (int i=0;i<filter_width;i++){
                for (int j=0;j<filter_width;j++){
                    int row_val = start_row + i;
                    int col_val = start_col + j;

                    float filter_val = d_filter[i*filter_width + j];

                
                    // If the filter overflows from the image area, multiply the out of bound
                    // coefficients with the nearest image pixel, hence changing the row_val
                    // and col_val values accordingly
                    row_val = (row_val <= 0) ? 0 : ((row_val >= num_rows) ? (num_rows-1) : row_val);
                    col_val = (col_val <= 0) ? 0 : ((col_val >= num_cols) ? (num_cols-1) : col_val);

                    // Calculating the index
                    int index = (row_val*num_cols) + col_val;

                    // Calculate the value for each channel based on the filter co-ordinates 
                    img_val_r = float(d_img[(index*3)]);
                    img_val_g = float(d_img[(index*3)+1]);
                    img_val_b = float(d_img[(index*3)+2]);

                    // Perform convolution
                    prod_r += (img_val_r * filter_val);
                    prod_g += (img_val_g * filter_val);
                    prod_b += (img_val_b * filter_val);


                }
            }

            // Writing result for each channel
            d_res[( ( (row * num_cols) + col) * 3)] = prod_r;
            d_res[( ( (row * num_cols) + col) * 3) + 1] = prod_g;
            d_res[( ( (row * num_cols) + col) * 3) + 2] = prod_b;

        }
        else{

            // For every row in the filter
            for (int i=0;i<filter_width;i++){

                // For every column in the filter
                for (int j=0;j<filter_width;j++){

                int temp_id_f = (((threadIdx.y -(filter_width/2) + i)*THREADSy + (threadIdx.x -(filter_width/2) + j))*3);
                float filter_val = d_filter[i*filter_width + j];

                // Calculate the value for each channel based on the filter co-ordinates 
                img_val_r = float(temp_img[temp_id_f]);
                img_val_g = float(temp_img[temp_id_f+1]);
                img_val_b = float(temp_img[temp_id_f+2]);

                // Perform convolution
                prod_r += (img_val_r * filter_val);
                prod_g += (img_val_g * filter_val);
                prod_b += (img_val_b * filter_val);
                }
            }
            
            // Writing result for each channel
            d_res[( ( (row * num_cols) + col) * 3)] = prod_r;
            d_res[( ( (row * num_cols) + col) * 3) + 1] = prod_g;
            d_res[( ( (row * num_cols) + col) * 3) + 2] = prod_b;
        }
    }
}

/*************************Pre-Processing Function*****************************/
void ImgBlur(Mat& h_img, Mat& h_res, float* filter, int filter_width){

        // Create events to time the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float exec_time;
        
        // Calculating the size of the frame for memory
        size_t img_size = (h_img.rows * h_img.cols * h_img.channels());

        // Calculating the size of the filter
        size_t filter_size = (filter_width * filter_width * sizeof(float));

        // Defining Device variables
        unsigned char *d_frame, *d_res;
        float* d_filter;

        // Allocating memory
        cudaMalloc((void**)&d_frame, img_size);
        cudaMalloc((void**)&d_res, img_size);
        cudaMalloc((void**)&d_filter, filter_size);

        // Copying frame to device
        cudaMemcpy(d_frame, h_img.ptr(), img_size, cudaMemcpyHostToDevice);

        // Copying filter from host to device
        cudaMemcpy(d_filter, filter, filter_size, cudaMemcpyHostToDevice);

        // Configuring threads and blocks
        int BLOCKSx = (h_img.cols + THREADSx -1) / THREADSx;
        int BLOCKSy = (h_img.rows + THREADSy -1) / THREADSy;

        dim3 threads(THREADSx, THREADSy);
        dim3 blocks(BLOCKSx, BLOCKSy);

        // Launching the kernel
        cudaEventRecord(start);
        ImgBlurKernel<<<blocks, threads>>>(d_frame, h_img.rows, h_img.cols,
                                           d_filter, filter_width, d_res);

        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();

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
        cudaFree(d_res);
        cudaFree(d_filter);

        return;
}

/************************Function to create Filter****************************/
void create_filter(float* h_filter){
      
  //now create the filter that they will use
  const int blurKernelWidth = kernel_width;
  const float blurKernelSigma = 2.;

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }
}



int main(void){

    // Reading the image
    Mat h_img = imread("../Images/cinque_terre_small.jpg", 1);
    Mat test_img = imread("../Images/cinque_terre.gold", 1);

    // Create the output image
    Mat h_res(h_img.rows, h_img.cols, CV_8UC3);

    // Calculate the filter size and allocate memory for it
    size_t filter_size = (kernel_width * kernel_width * sizeof(float));
    float* h_filter = (float*)malloc(filter_size);

    // Call the create_filter to fill the filter
    create_filter(h_filter);

    // Call the pre-processing function, that further calls the kernel
    ImgBlur(h_img, h_res, h_filter, kernel_width);

    // Creating the difference image
    Mat diff_img = abs(test_img - h_res);

    // Calculate the difference based on the in-built function in OpenCV
    Mat ref_img;
    cv::GaussianBlur(h_img, ref_img, Size2i(kernel_width, kernel_width), 0);
    Mat diff_opencv = abs(ref_img - h_res);

    // Show the results
    cv::imshow("Original", h_img);
    cv::imshow("Blurred", h_res);
    cv::imshow("Ref Image", test_img);
    cv::imshow("Diff", diff_img);
    cv::imshow("CV-Blur", ref_img);
    cv::imshow("diff-CV", diff_opencv);

    // Saving the result to disk
    cv::imwrite("../Images/Blurred_Output_final_shared.png", h_res);
    cv::imwrite("../Images/Gaussian_Blur_Output.png", ref_img);

    // Wait for any keystrokes
    cv::waitKey(0);

    // Free the memory allocated to the filter
    delete[] h_filter;

    return 0;
}

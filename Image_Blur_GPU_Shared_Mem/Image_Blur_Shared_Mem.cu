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

/*****************************Convolution Kernel******************************/
__global__ void ImgBlurKernel(unsigned char* d_img, int num_rows, int num_cols,
                              float* d_filter, int filter_width, 
                              unsigned char* d_res){
    
  // Calculating the row and column indices
  int row = ((threadIdx.y * 2)  + (blockDim.y * blockIdx.y));
  int col = ((threadIdx.x * 2) + (blockDim.x * blockIdx.x));

  // Declaring shared memory
  __shared__ unsigned char temp_img[(BLOCK_PADDED - 1)*(BLOCK_PADDED - 1)*3];
      
        // Case 1
        int start_row = row - filter_width/2;
        int start_col = col - filter_width/2;

        int img_idx = (start_col);
        int img_idy = (start_row);

        int temp_idx = (threadIdx.x * 2);
        int temp_idy = (threadIdx.y * 2);

        int temp_id_f = ((temp_idy * (BLOCK_PADDED-1) + temp_idx)*3);
        int img_id_f = ((img_idy*num_cols + img_idx)*3);

        int x_n = 0, y_n = 0;

        if (start_row < 0 || start_col < 0 || start_row > num_rows -1 || start_col > num_cols -1){

        y_n = (start_row < 0) ? 0 : ((start_row >= num_rows) ? (num_rows-1) : (start_row));
        x_n = (start_col < 0) ? 0 : ((start_col >= num_cols) ? (num_cols-1) : (start_col));
    

        img_id_f = ((y_n*num_cols + x_n)*3); 

        temp_img[temp_id_f] = d_img[img_id_f];
        temp_img[temp_id_f + 1] = d_img[img_id_f + 1];
        temp_img[temp_id_f + 2] = d_img[img_id_f + 2];

        }
        else{
            temp_img[temp_id_f] = d_img[img_id_f];
            temp_img[temp_id_f + 1] = d_img[img_id_f + 1];
            temp_img[temp_id_f + 2] = d_img[img_id_f + 2];
        }

        // Case 2
        img_idx = start_col;
        img_idy = start_row + 1;

        temp_idx = (threadIdx.x * 2);
        temp_idy = (threadIdx.y * 2) + 1;

        temp_id_f = ((temp_idy * (BLOCK_PADDED-1) + temp_idx)*3);
        img_id_f = ((img_idy*num_cols + img_idx)*3);
        
        if ((start_row+1) > num_rows-1 || start_col < 0 || (start_row+1) < 0 || start_col > num_cols -1){

        y_n = (start_row < 0) ? 0 : ((start_row >= num_rows) ? (num_rows-1) : (start_row+1));
        x_n = (start_col < 0) ? 0 : ((start_col >= num_cols) ? (num_cols-1) : (start_col));

        img_id_f = ((y_n*num_cols + x_n)*3); 

        temp_img[temp_id_f] = d_img[img_id_f];
        temp_img[temp_id_f + 1] = d_img[img_id_f + 1];
        temp_img[temp_id_f + 2] = d_img[img_id_f + 2];

        }
        else{
            temp_img[temp_id_f] = d_img[img_id_f];
            temp_img[temp_id_f + 1] = d_img[img_id_f + 1];
            temp_img[temp_id_f + 2] = d_img[img_id_f + 2];
        }

        // Case 3
        img_idx = start_col + 1;
        img_idy = start_row;

        temp_idx = (threadIdx.x * 2) + 1;
        temp_idy = (threadIdx.y * 2);

        temp_id_f = ((temp_idy * (BLOCK_PADDED-1) + temp_idx)*3);
        img_id_f = ((img_idy*num_cols + img_idx)*3);

        if ((start_col+1) > num_cols -1 || start_row < 0 || (start_col+1) < 0 || start_row > num_rows - 1){

        y_n = (start_row < 0) ? 0 : ((start_row >= num_rows) ? (num_rows-1) : start_row);
        x_n = (start_col < 0) ? 0 : ((start_col >= num_cols) ? (num_cols-1) : start_col+1);

        img_id_f = ((y_n*num_cols + x_n)*3); 

        temp_img[temp_id_f] = d_img[img_id_f];
        temp_img[temp_id_f + 1] = d_img[img_id_f + 1];
        temp_img[temp_id_f + 2] = d_img[img_id_f + 2];

        }
        else{
            temp_img[temp_id_f] = d_img[img_id_f];
            temp_img[temp_id_f + 1] = d_img[img_id_f + 1];
            temp_img[temp_id_f + 2] = d_img[img_id_f + 2];
        }

        // Case 4
        img_idx = start_col + 1;
        img_idy = start_row + 1;

        temp_idx = (threadIdx.x * 2) + 1;
        temp_idy = (threadIdx.y * 2) + 1;

        temp_id_f = ((temp_idy * (BLOCK_PADDED-1) + temp_idx)*3);
        img_id_f = ((img_idy*num_cols + img_idx)*3);

        if ((start_row+1) > num_rows -1 || (start_col+1) > num_cols - 1 || (start_col+1) < 0 || (start_row+1) < 0){
 
        y_n = (start_row < 0) ? 0 : ((start_row >= num_rows) ? (num_rows-1) : start_row+1);
        x_n = (start_col < 0) ? 0 : ((start_col >= num_cols) ? (num_cols-1) : start_col+1);

        img_id_f = ((y_n*num_cols + x_n)*3); 

        temp_img[temp_id_f] = d_img[img_id_f];
        temp_img[temp_id_f + 1] = d_img[img_id_f + 1];
        temp_img[temp_id_f + 2] = d_img[img_id_f + 2];

        }
        else{
            temp_img[temp_id_f] = d_img[img_id_f];
            temp_img[temp_id_f + 1] = d_img[img_id_f + 1];
            temp_img[temp_id_f + 2] = d_img[img_id_f + 2];
        }


    // Barrier for all threads
    __syncthreads();


    //*************************************************************************

    //Performing convolution

    // Initializing temporary product variable

    col = (threadIdx.x + (blockDim.x*blockIdx.x));
    row = (threadIdx.y + (blockDim.y*blockIdx.y));

    if (row < num_rows && col < num_cols){

        float prod_r = 0.0f, prod_g = 0.0f, prod_b = 0.0f;

        float img_val_r = 0, img_val_g = 0, img_val_b = 0;

        int temp_id_conv = 0;
        float filter_val = 0.0f;
        
        for (int i=0;i<filter_width;i++){
            for (int j=0;j<filter_width;j++){

                temp_id_conv = (((threadIdx.y + i)*(BLOCK_PADDED-1) + (threadIdx.x + j))*3);
                
                filter_val = d_filter[i*filter_width + j];
                
                img_val_r = float(temp_img[temp_id_conv]);
                img_val_g = float(temp_img[temp_id_conv + 1]);
                img_val_b = float(temp_img[temp_id_conv + 2]);

                prod_r += (img_val_r * filter_val);
                prod_g += (img_val_g * filter_val);
                prod_b += (img_val_b * filter_val);
            }
        }


        int idx = ((row*num_cols + col)*3);

        d_res[idx] = prod_r;
        d_res[idx + 1] = prod_g;
        d_res[idx + 2] = prod_b;

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

    //Mat h_img = imread("../Images/index.jpeg", 1);
    //Mat h_img = imread("../Images/cinque_terre_small.jpg", 1);
    //Mat h_img = imread("/usr/share/backgrounds/brad-huchteman-stone-mountain.jpg", 1);
    //Mat h_img = imread("../Images/138728.jpg", 1);
    //Mat h_img = imread("../Images/2040735.jpg", 1);
    Mat h_img = imread("../Images/UI-Sidewalk-640x480.jpg", 1);


    //resize(h_img, h_img, Size(280, 184));
    //resize(h_img, h_img, Size(560, 320));
    cout << h_img.cols << " || " << h_img.rows << endl;

    // Resizing the image
    if (h_img.cols % 8 != 0 && h_img.rows % 8 != 0){
        resize(h_img, h_img, Size((h_img.cols/THREADSx + 1)*THREADSx, (h_img.rows/THREADSy + 1)*THREADSy));
    }
    cout << h_img.cols << " || " << h_img.rows << endl;

    // Create the output image
    Mat h_res(h_img.rows, h_img.cols, CV_8UC3);

    // Calculate the filter size and allocate memory for it
    size_t filter_size = (kernel_width * kernel_width * sizeof(float));
    float* h_filter = (float*)malloc(filter_size);

    // Call the create_filter to fill the filter
    create_filter(h_filter);

    // Call the pre-processing function, that further calls the kernel
    ImgBlur(h_img, h_res, h_filter, kernel_width);

    // Calculate the difference based on the in-built function in OpenCV
    Mat ref_img;
    cv::GaussianBlur(h_img, ref_img, Size2i(kernel_width, kernel_width), 0);
    Mat diff_opencv = abs(ref_img - h_res);

    // Show the results
    cv::imshow("Original", h_img);
    cv::imshow("Blurred", h_res);
    cv::imshow("CV-Blur", ref_img);
    cv::imshow("diff-CV", diff_opencv);

    // Saving Images to disk
    imwrite("../Images/Blurred_Shared_mem_final_fruits.png", h_res);
    imwrite("../Images/Difference_Image_Shared_mem.png", diff_opencv);

    // Wait for any keystrokes
    cv::waitKey(0);

    // Free the memory allocated to the filter
    delete[] h_filter;

    return 0;
}

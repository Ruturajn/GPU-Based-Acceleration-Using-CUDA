#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

__global__ void ImageInversionKernel(unsigned char* d_img, int num_rows, int num_cols,
                            int channels, unsigned char* d_res){
    
    // Declaring the Indices
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);

    // Creating the index to access each pixel value which has RGB values
    int index = (row*num_cols) + col; 

    // Performing the Inversion Operation
    if (row < num_rows && col < num_cols){
        for (int i=0;i<channels;i++){
            // Inverting the values of each pixel's R, G and B.
            d_res[(index*channels) + i] = 255 - d_img[(index*channels) + i];
        }
    }
}

void ImageInv(Mat &frame, Mat &h_res){

        // Calculating the size of the frame for memory
        size_t img_size = (frame.rows * frame.cols * frame.channels());

        // Definning Device variables
        unsigned char *d_frame, *d_res;

        // Allocating memory
        cudaMalloc((void**)&d_frame, img_size);
        cudaMalloc((void**)&d_res, img_size);

        // Copying frame to device
        cudaMemcpy(d_frame, frame.ptr(), img_size, cudaMemcpyHostToDevice);

        // Configuring threads and blocks
        int THREADSx = 16;
        int THREADSy = 16;

        dim3 threads(THREADSx, THREADSy);
        dim3 blocks(((frame.cols + THREADSx - 1)/THREADSx), (((frame.rows + THREADSy - 1)/THREADSy)));

        // Launching the kernel
        ImageInversionKernel<<<blocks, threads>>>(d_frame, frame.rows, frame.cols, frame.channels(), d_res);

        // Copying the results back to host
        cudaMemcpy(h_res.ptr(), d_res, img_size, cudaMemcpyDeviceToHost);

        // Free the allocated memory
        cudaFree(d_frame);
        cudaFree(d_res);

        return;
}

int main(void){

    // Starting the video capture from the Webcam
    VideoCapture cap(0);

    // Check whether webcam started
    if (cap.isOpened() == false){
        cout << "Error opening Webcam" << endl;
        return -1;
    }

    while(true){
        // Creating a frame
        Mat frame;

        // Reading a frame from the webcam input
        bool flag = cap.read(frame);

        // Creating the result image on host
        Mat h_res(frame.rows, frame.cols, CV_8UC3);

        // Calling the host function that further calls the kernel
        ImageInv(frame, h_res);

        // Showing the results
        imshow("Input", frame);
        imshow("Inverted", h_res);

        // Waiting 1ms for the user to press 'q'
        if (waitKey(1) == 'q')
            break;

    }

    return 0;
}

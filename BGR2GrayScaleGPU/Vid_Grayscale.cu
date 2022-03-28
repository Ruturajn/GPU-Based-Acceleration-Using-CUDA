#include <iostream>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define R_Coeff 0.299f
#define G_Coeff 0.587f
#define B_Coeff 0.114f

using namespace std;
using namespace cv;

/*****************************************************************************/
// Kernel for converting Image from Color to Grayscale
__global__ void BGR2GrayKernel(unsigned char* d_img, int num_rows, int num_cols, 
                         int channels, unsigned char* d_res){
        
    // Calculating row and col indices
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    int row = threadIdx.y + (blockIdx.y * blockDim.y);

    // Calculating the index for the 2D image
    int index = (row*num_cols) + col;

    // Conversion from color to grayscale
    if (row < num_rows && col < num_cols){
        d_res[index] = (unsigned char)((B_Coeff * d_img[(index*channels)]) +
                                  (G_Coeff * d_img[(index*channels)+1]) +
                                  (R_Coeff * d_img[(index*channels)+2]));
    } 

}
/*****************************************************************************/


/*****************************************************************************/
// Host funtion for preprocessing and calling the kernel
void BGR2Gray(Mat& h_img, Mat& h_res){
    // Calculating the size of the image
    size_t in_img_size = (h_img.rows * h_img.cols * h_img.channels());
    size_t out_img_size = (h_img.rows * h_img.cols);

    // Creating device variales
    unsigned char *d_img, *d_res;

    // Allocating Memory for device variables
    cudaMalloc((void**)&d_img, in_img_size);
    cudaMalloc((void**)&d_res, out_img_size);

    // Copying the img from host to device
    cudaMemcpy(d_img, h_img.ptr(), in_img_size, cudaMemcpyHostToDevice);

    // Configuring the threads and blocks
    int THREADSx = 16;
    int THREADSy = 16;

    int BLOCKSx = (h_img.cols + THREADSx -1) / THREADSx;
    int BLOCKSy = (h_img.rows + THREADSy -1) / THREADSy;

    dim3 threads(THREADSx, THREADSy);
    dim3 blocks(BLOCKSx, BLOCKSy);

    // Calling the Kernel
    BGR2GrayKernel<<<blocks, threads>>>(d_img, h_img.rows, h_img.cols,
                                        h_img.channels(), d_res);

    // Copying the result back to host
    cudaMemcpy(h_res.ptr(), d_res, out_img_size, cudaMemcpyDeviceToHost);

    // Free the allocated memory
    cudaFree(d_img);
    cudaFree(d_res);

    return;
}
/*****************************************************************************/


int main(void){

    // Instantiating Video Capture
    VideoCapture cap(0);

    // Checking whether the webcam started
    if (cap.isOpened() == false){
        cout << "Error Opening Webcam" << endl;
        return -1;
    }

    while(true){
        // Creating frame 
        Mat frame;

        // Reading the frame
        cap.read(frame);

        // Creating the resulting image
        Mat h_res(frame.rows, frame.cols, CV_8UC1);

        // Calling the function that further calls the kernel
        BGR2Gray(frame, h_res);

        // Showing the results
        imshow("Color Input", frame);
        imshow("Grayscale Output", h_res);

        // Waiting for 1ms after each frame for the user to press 'q'
        if (waitKey(1)== 'q')
            break;
    }
    return 0;
}

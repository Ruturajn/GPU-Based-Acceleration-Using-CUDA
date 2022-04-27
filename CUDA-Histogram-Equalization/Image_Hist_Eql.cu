#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <assert.h>

using namespace cv;
using namespace std;

#define THREADSx 16
#define THREADSy 16
#define num_bins 1024

/************************* Scan Compute Kernel ******************************/
__global__ void scanKernel(unsigned int *d_in, int end_ind, unsigned int *d_out)
{
    /**
     * Implementation of the Hillis Steele Scan, which is an inclusive scan,
     * but the CDF requires an exclusive scan. Hence here, we will look at
     * a variation of the above algorithm that conatins some tweaks to make
     * it an exclusive scan.
     */

    // Find the global index for the array
    int glob_index = threadIdx.x + (blockIdx.x * blockDim.x);

    // Make sure we don't access stuff that is out of memory
    if (glob_index < num_bins)
    {
        // For hopping through the array, with a stride that increases by
        // a factor of 2 every iteration.
        // Refer link : https://youtu.be/RdfmxfZBHpo

        for (unsigned int hop = 1; hop <= end_ind / 2; hop *= 2)
        {
            // Get the value of the element to the left based on the number
            // hops.
            int glob_hop = glob_index - hop;

            // Make sure the index is not negative
            if (glob_hop >= 0)
            {
                // Read the value to the left of the current element based on
                // the number of hops from global memory
                unsigned int val = d_in[glob_hop];

                // Barrier for threads
                __syncthreads();

                // Perform the addition operation
                d_in[glob_index] += val;

                // Barrier for threads
                __syncthreads();
            }
        }
        //Place the result in the output shifted by 1 to the right
        if ((glob_index + 1) < num_bins)
            d_out[glob_index+1] = d_in[glob_index]; 
    }
}

/********************* Histogram Computation Kernel *************************/
__global__ void ImageHistKernel(float *d_lumin, float logmin, float logmax,
                                int num_rows, int num_cols, unsigned int *d_hist,
                                float range_val, unsigned int *d_cdf)
{

    // Calculating the row and column
    // int tid = (threadIdx.x + (blockDim.x * blockIdx.x));
    int row = (threadIdx.y + (blockDim.y * blockIdx.y));
    int col = (threadIdx.x + (blockDim.x * blockIdx.x));

    if (row < num_rows && col < num_cols)
    {
        // Getting global image index
        int g_ind = row * num_cols + col;

        // Determine the bin to which this threadId will write
        unsigned int bin_index = (unsigned int)((d_lumin[g_ind] - logmin) / range_val * num_bins);
        unsigned int bin = min((num_bins - 1), bin_index);

        // Increment the respective bin of the histogram
        atomicAdd(&(d_hist[bin]), 1);

        __syncthreads();
    }
}

/******************* Histogram Verification Function *************************/
void referenceCalculation(const float *const h_logLuminance, unsigned int *const h_cdf,
                          const size_t numRows, const size_t numCols, const size_t numBins,
                          float &logLumMin, float &logLumMax, unsigned int *h_hist)
{
    logLumMin = h_logLuminance[0];
    logLumMax = h_logLuminance[0];

    // Step 1
    // first we find the minimum and maximum across the entire image
    for (size_t i = 1; i < numCols * numRows; ++i)
    {
        logLumMin = std::min(h_logLuminance[i], logLumMin);
        logLumMax = std::max(h_logLuminance[i], logLumMax);
    }

    // Step 2
    float logLumRange = logLumMax - logLumMin;

    // Step 3
    // next we use the now known range to compute
    // a histogram of numBins bins
    unsigned int *histo = new unsigned int[numBins];

    for (size_t i = 0; i < numBins; ++i)
        histo[i] = 0;

    for (size_t i = 0; i < numCols * numRows; ++i)
    {
        unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                                    static_cast<unsigned int>((h_logLuminance[i] - logLumMin) / logLumRange * numBins));
        histo[bin]++;
    }

    // for (int i=0;i<num_bins;i++){
    //     if (histo[i] != h_hist[i]){
    //         cout << "Element not matching at : " << i << endl;
    //         cout << "histo[" << i << "] = " << histo[i] << " || " << "h_hist[" << i << "] = " << h_hist[i] << endl;
    //         cout << "---------------------------------------------------------------------------------" << endl;

    //     }
    // }

    // Step 4
    // finally we perform and exclusive scan (prefix sum)
    // on the histogram to get the cumulative distribution
    h_cdf[0] = 0;
    for (size_t i = 1; i < numBins; ++i)
    {
        h_cdf[i] = h_cdf[i - 1] + histo[i - 1];
    }

    delete[] histo;
}

/************************ Pre-Processing Function ****************************/
void ImageHist(Mat &h_img, Mat &h_res)
{

    // Converting the Image to YUV colour space
    cv::cvtColor(h_img, h_img, COLOR_BGR2YUV);

    // Convert the image to single-precision floating point
    h_img.convertTo(h_img, CV_32FC3, 1.0f,0.0001f);
    size_t channel_size = (h_img.rows * h_img.cols * sizeof(float));

    // Creating separate image for intensity channel
    Mat Y_chn(h_img.rows, h_img.cols, CV_32FC1),
        U_chn(h_img.rows, h_img.cols, CV_32FC1),
        V_chn(h_img.rows, h_img.cols, CV_32FC1);

    extractChannel(h_img, Y_chn, 0);
    extractChannel(h_img, U_chn, 1);
    extractChannel(h_img, V_chn, 2);

    // Y_chn.convertTo(Y_chn, CV_32FC1);
    // U_chn.convertTo(U_chn, CV_32FC1);
    // V_chn.convertTo(V_chn, CV_32FC1);

    // Calculating the min and max log values in the luminance channel
    float log_min = 0.0f, log_max = 0.0f, range_val = 0.0f;
    float *Y_chn_data = (float *)Y_chn.ptr();

    for (int i = 0; i < (h_img.rows * h_img.cols); i++)
    {

        if (Y_chn_data[i] > 0)
        {
            Y_chn_data[i] = log10f(Y_chn_data[i]);
            //cout << "Y_chn_data[" << i << "] = " << Y_chn_data[i] << endl;
        }

        if (Y_chn_data[i] < log_min)
            log_min = Y_chn_data[i];

        if (Y_chn_data[i] > log_max)
            log_max = Y_chn_data[i];
    }

    range_val = log_max - log_min;

    // Initializing our histogram bins to 0
    unsigned int h_hist[num_bins] = {0};
    unsigned int h_cdf[num_bins] = {0};
    unsigned int h_cdf_ref[num_bins] = {0};

    // Creating device variables
    float *d_lumin;
    unsigned int *d_hist, *d_cdf, *d_cdf_temp;

    // Allocating Memory for device variables
    cudaMalloc((void **)&d_lumin, channel_size);
    cudaMalloc((void **)&d_cdf, sizeof(unsigned int) * num_bins);
    cudaMalloc((void **)&d_hist, sizeof(unsigned int) * num_bins);
    cudaMalloc((void **)&d_cdf_temp, sizeof(unsigned int) * num_bins);

    // Copy the channel to the device
    cudaMemcpy(d_lumin, Y_chn_data, channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, h_hist, num_bins * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Calculating the block size
    int BLOCKSx = ((h_img.cols + THREADSx - 1) / THREADSx);
    int BLOCKSy = ((h_img.rows + THREADSy - 1) / THREADSy);
    // int BLOCKSx = 1;

    dim3 threads(THREADSx, THREADSy);
    dim3 blocks(BLOCKSx, BLOCKSy);

    // Call the Histogram kernel
    ImageHistKernel<<<blocks, threads>>>(d_lumin, log_min, log_max,
                                         h_img.rows, h_img.cols,
                                         d_hist, range_val, d_cdf);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Copy the results back to host
    cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * num_bins, cudaMemcpyDeviceToHost);

    //scanKernel<<<((num_bins + THREADSx - 1)/THREADSx), THREADSx>>>(d_hist, num_bins, d_cdf);
    scanKernel<<<num_bins,num_bins>>>(d_hist, num_bins, d_cdf);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaMemcpy(h_cdf, d_cdf, sizeof(unsigned int) * num_bins, cudaMemcpyDeviceToHost);

    // Checking whether histogram calculation is correct
    referenceCalculation(Y_chn_data, h_cdf_ref, h_img.rows, h_img.cols,
                         num_bins, log_min, log_max, h_hist);

    // for (int i = 0; i < num_bins; i++)
    // {
    //     if (h_cdf[i] != h_cdf_ref[i])
    //     {
    //         cout << "Element not matching at : " << i << endl;
    //         cout << "h_cdf[" << i << "] = " << h_cdf[i] << " || "
    //              << "h_cdf_ref[" << i << "] = " << h_cdf_ref[i] << endl;
    //         cout << "---------------------------------------------------------------------------------" << endl;
    //     }
    // }

    // Do Tone mapping based on equalised histogram

        // Normalize the cdf
        float norm_factor = 1.0f/h_cdf[num_bins-1];
        float h_norm_cdf[num_bins];
        for (int i=0;i<num_bins;i++){
            h_norm_cdf[i] = h_cdf[i] * norm_factor;
            //cout << "h_norm_cdf[i] is : " << h_norm_cdf[i] << endl;
        }
        
        // Find the min of the CDF
        // unsigned int cdf_min = 0;
        // for (int i=0;i<num_bins;i++){
        //     if (cdf_min > h_cdf[i])
        //         cdf_min = h_cdf[i];
        // }

        // Perform the actual tone mapping
        for (int i=0;i<(h_img.rows * h_img.cols);i++){
            int bin = (int)((num_bins * (Y_chn_data[i] - log_min))/range_val);
            // if (Y_chn_data[i] == 0)
            //     cout << "bin = " << bin << endl;
            int bin_ind = min(num_bins-1,bin);
            //cout << "bin id is : " << bin_ind << endl;
            Y_chn_data[i] = h_norm_cdf[bin_ind];
        }
    
    // Converting the Y_chn_data back to an image
    Mat Y_chn_eql(h_img.rows, h_img.cols, CV_32FC1, Y_chn_data);

    // Reconstructing the image
    Mat merged[3] = {Y_chn_eql, U_chn, V_chn};
    cv::merge(merged, 3, h_res);

    // Convert Image back to BGR
    cv::cvtColor(h_res, h_res, COLOR_YUV2BGR);
    h_res.convertTo(h_res, CV_8UC3);

    // Free the allocated memory
    cudaFree(d_lumin);
    cudaFree(d_cdf);
    cudaFree(d_hist);
    cudaFree(d_cdf_temp);
}

/***************************** The Main Function *****************************/
int main(void)
{

    // Read the Input Image
    Mat h_img = imread("../Images/memorial_raw.png", 1);
    //imshow("Original", h_img);

    // Creating the result image
    Mat h_res(h_img.rows, h_img.cols, CV_32FC3);

    // Call the pre-processing function that will further call the kernel
    ImageHist(h_img, h_res);

    // Reading the reference image and determining the difference
    Mat h_ref = imread("../Images/memorial_png.gold", 1);
    //Mat h_diff = abs(h_ref - h_res);

    //h_img.convertTo(h_img,CV_8UC3);

    //Showing the results
    //imshow("Original", h_img);
    imshow("Hist-Eql", h_res);
    //imshow("Diff", h_diff);
    //imshow("Ref-Img", h_ref);

    //imwrite("../Images/Hist-Eql.png", h_res);

    // Wait indefinitely for any keystrokes
    waitKey(0);

    return 0;
}
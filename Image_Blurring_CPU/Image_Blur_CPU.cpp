#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <cassert>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#define kernel_width 9

void channel_conv(unsigned char* channel, unsigned char* ch_blur,
                  int num_rows, int num_cols, float* filter, 
                  int filter_width){
  //Dealing with an even width filter is trickier
  assert(filter_width % 2 == 1);

  //For every pixel in the image
  for (int r = 0; r < (int)num_rows; ++r) {
    for (int c = 0; c < (int)num_cols; ++c) {
      float result = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filter_width/2; filter_r <= filter_width/2; ++filter_r) {
        for (int filter_c = -filter_width/2; filter_c <= filter_width/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
	  int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(num_rows - 1));
          int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(num_cols - 1));
                    
          float image_value = static_cast<float>(channel[image_r * num_cols + image_c]);
          float filter_value = filter[(filter_r + filter_width/2) * filter_width + filter_c + filter_width/2];
          result += image_value * filter_value;
          
          /*************************** Alternate Way of doing the things above *******************************/
          // int image_r = r + filter_r;
          // int image_c = c + filter_c;
          
          // image_r = (image_r <= 0) ? 0 : ((image_r >= num_rows) ? (num_rows-1) : image_r);
          // image_c = (image_c <= 0) ? 0 : ((image_c >= num_cols) ? (num_cols-1) : image_c);

          // float image_value = static_cast<float>(channel[image_r * num_cols + image_c]);
          // float filter_value = filter[(filter_r + filter_width/2) * filter_width + filter_c + filter_width/2];
          // result += image_value * filter_value;
          /****************************************************************************************************/

        }
      }

      ch_blur[r * num_cols + c] = (unsigned char)result;
    }
  }
}

void ref_calc(Mat& in_img, Mat& out_img, int num_rows, int num_cols,
              float* filter, int filter_width){
    
  // Create separate images for eac channel
  Mat ch_red, ch_green, ch_blue;

  // Extract the channels from the input image
  extractChannel(in_img, ch_red, 0);
  extractChannel(in_img, ch_green, 1);
  extractChannel(in_img, ch_blue, 2);

  // Extract the data from each of the channels in form of
  // unsigned char*
  unsigned char* chr_data = ch_red.data;
  unsigned char* chg_data = ch_green.data;
  unsigned char* chb_data = ch_blue.data;

  // Creat arrays for the blurred output from each channel
  unsigned char* red_blur = new unsigned char[num_cols*num_rows];
  unsigned char* green_blur = new unsigned char[num_cols*num_rows];
  unsigned char* blue_blur = new unsigned char[num_cols*num_rows];

  // Call the channel_conv function for image blurring
  channel_conv(chr_data, red_blur, num_rows, num_cols, filter, filter_width);
  channel_conv(chg_data, green_blur, num_rows, num_cols, filter, filter_width);
  channel_conv(chb_data, blue_blur, num_rows, num_cols, filter, filter_width);

  // Reconstructing the image from the blurred channels
  Mat ch_red_reconstruct(in_img.rows, in_img.cols, CV_8UC1, red_blur);
  Mat ch_green_reconstruct(in_img.rows, in_img.cols, CV_8UC1, green_blur);
  Mat ch_blue_reconstruct(in_img.rows, in_img.cols, CV_8UC1, blue_blur);

  // Creating an array containing all the 3 images from above
  Mat merged_img[3] = {ch_red_reconstruct, ch_green_reconstruct, ch_blue_reconstruct};

  // Merging the image to get the final output image
  merge(merged_img, 3, out_img);
  
  delete[] red_blur;
  delete[] green_blur;
  delete[] blue_blur;
}

void create_filter(float* h_filter){
  
  //now create the filter that they will use
  const int blurKernelWidth = 9;
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

  // cout << "Normal Factor : " << normalizationFactor << endl;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }
}


int main(void){

  // Read the Image
  Mat img = imread("./cinque_terre_small.jpg", 1);

  // Create the result image
  Mat blurred_img;

  // Read the reference image
  Mat test_img = imread("./cinque_terre.gold", 1);

  // Calculate the filter size and allocate memory for it
  size_t filter_size = kernel_width * kernel_width * sizeof(float);
  float *h_filter = (float*)malloc(filter_size);

  // Call the create_filter function to fill the filter
  create_filter(h_filter);

  // Call the ref_calc function for pre-processing which further calls the kernel
  ref_calc(img, blurred_img, img.rows, img.cols, h_filter, kernel_width);

  // Creating a differnce image based on the computed result and the reference image
  Mat diff = abs(test_img - blurred_img);

  // Calculate the difference based on the in-built function in OpenCV
  Mat ref_img;
  GaussianBlur(img, ref_img, Size2i(kernel_width, kernel_width), 0);
  Mat diff_opencv = abs(ref_img - blurred_img);

  // Show the results
  imshow("Original", img);
  imshow("Blurred", blurred_img);
  imshow("Ref Image", test_img);
  imshow("diff", diff);
  imshow("CV-Blur", ref_img);
  imshow("diff-CV", diff_opencv);

  // Wait for any keystrokes
  waitKey(0);

  // Free the allocated memory
  delete[] h_filter;
 
  return 0;
}

# This Folder conatins code for Laplacian Filtering Implementation using Opencv integrated with CUDA, run on NVIDIA GeForce 940MX GPU with a Compute Capability of 5.0.

The Laplacian, Scharr and Sobel Filtering Methods are generally used for Edge Detection in Images and Videos. The Scharr and Sobel methods do not parallely compute the edges in the horizontal and vertical direction, hence the edges for horizontal and vertical direction need to be extracted and manually addded for complete edge detection. This drawback is absent in Laplacian Filtering Technique, but it is extremely sensitive to noise, hence denoising operation needs to be done prior to edge detection. In this code, Gaussian Filtering is used for denoising.

### For executing the code OpenCV needs to be built with CUDA Support.

### Execution of the Program (In Linux based OS):

```
$ cd <path/to/location/of/the/program/>
$ make -j($nproc)
```

# Image Blurring Operation on the CPU

This code blurs an input image by performing a convolution of the image with a given filter.

## Execution of the Code (In a Linux based OS):

```
$ cd </path/to/the/location/of/the/program>
$ g++ -std=c++11 Image_Blur_CPU.cpp `pkg-config --cflags --libs opencv` -o test.out
$ ./test.out
```

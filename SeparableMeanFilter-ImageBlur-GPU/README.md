# Image Blurring Operation on Images using CUDA and OpenCV with Separable Filters

This code blurs an input image by convolving it with a given filter. All of this is done by harnessing the GPU compute capability using CUDA. Here, we convolve the images with a row and a column filter instead of a 2D filter, which reduces computations from nxn to n+n, hence increasing performance.

## Execution of the Code (In a Linux based OS):

```
$ cd </path/to/the/location/of/the/program>
$ make
$ ./one.out
```

**Note: You might want to change the architecture specific flags in the Makefile if your GPU has a different compute and sm architecture. Also, there might be a need to change the path for the input image while reading it.**

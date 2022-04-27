# GPU-Based-Acceleration-Using-CUDA

GPU-Based-Acceleration-Using-CUDA

```
./
├── 2D-Matrix-Add
│   ├── Add_Matrix.cu
│   ├── Add_Matrix_v2.cu
│   ├── Makefile
│   └── README.md
├── BGR2GrayScaleGPU
│   ├── Makefile
│   ├── README.md
│   └── Vid_Grayscale.cu
├── CD-Classifier
│   ├── Data.m
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   ├── CNNWeights_Layer3_128.h
│   │   └── CNNWeights_Layer4_1.h
│   ├── Input_Image.txt
│   ├── Makefile
│   ├── Model_CD.m
│   ├── src
│   │   ├── CNN_Inference.cu
│   │   ├── CNN_Layers.cu
│   │   ├── main.cpp
│   │   └── Pre_Process.cu
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       ├── Dense_Layer_128_Bias.txt
│       ├── Dense_Layer_128_Weights.txt
│       └── Dense_Layer_1_Weights.txt
├── CNN-Accel
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   └── CNNWeights_Layer3.h
│   ├── Makefile
│   ├── src
│   │   ├── CNN_Inference.cu
│   │   ├── main.cpp
│   │   └── Pre_Process.cu
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       └── Dense_Layer_Weight.txt
├── CNN-Accel-CPP
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   └── CNNWeights_Layer3.h
│   ├── Makefile
│   ├── src
│   │   ├── CNN_Inference.cpp
│   │   ├── main.cpp
│   │   └── Pre_Process.cpp
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       └── Dense_Layer_Weight.txt
├── CNN-Accel-Test
│   ├── inc
│   │   ├── CNN_Funcs.h
│   │   ├── CNNWeights_Layer1.h
│   │   ├── CNNWeights_Layer2.h
│   │   └── CNNWeights_Layer3.h
│   ├── Makefile
│   ├── src
│   │   ├── CNN_Inference.cu
│   │   ├── main.cpp
│   │   └── Pre_Process.cu
│   └── Text-Weight-Files
│       ├── Bias_Layer_1.txt
│       ├── Bias_Layer_2.txt
│       ├── Conv_Layer_1.txt
│       ├── Conv_Layer_2.txt
│       └── Dense_Layer_Weight.txt
├── CUDA-Histogram-Equalization
│   ├── Image_Hist_Eql_8_bit.cu
│   ├── Image_Hist_Eql.cu
│   └── Makefile
├── Image_Blur_GPU
│   ├── cinque_terre.gold
│   ├── cinque_terre_small.jpg
│   ├── Image_Blur.cu
│   ├── Makefile
│   └── README.md
├── Image_Blur_GPU_Alt
│   ├── cinque_terre.gold
│   ├── cinque_terre_small.jpg
│   ├── Image_Blur_Alt.cu
│   ├── Makefile
│   └── README.md
├── Image_Blur_GPU_Shared_Global_Mem
│   ├── cinque_terre.gold
│   ├── cinque_terre_small.jpg
│   ├── Image_Blur_GPU_Shared_Global_Mem.cu
│   ├── Makefile
│   └── README.md
├── Image_Blur_GPU_Shared_Mem
│   ├── 138728.jpg
│   ├── 2040735.jpg
│   ├── cinque_terre_small.jpg
│   ├── Image_Blur_Shared_Mem.cu
│   ├── index.jpeg
│   ├── Makefile
│   ├── README.md
│   └── UI-Sidewalk-640x480.jpg
├── Image_Blurring_CPU
│   ├── cinque_terre.gold
│   ├── cinque_terre_small.jpg
│   ├── Image_Blur_CPU.cpp
│   └── README.md
├── ImageInversionGPU
│   ├── Makefile
│   ├── README.md
│   └── Vid_Image_Inversion.cu
├── LaplacianFilteringGPU
│   ├── Makefile
│   ├── README.md
│   └── Vid_Edge.cpp
├── PCD-Operations
│   ├── airplane.ply
│   ├── ant.ply
│   ├── Basic-PCD-IO
│   │   ├── Makefile
│   │   ├── Matrix_Transform.cpp
│   │   ├── Passthrough-Filtering
│   │   │   ├── CMakeLists.txt
│   │   │   └── passthrough.cpp
│   │   ├── Test_IO.cpp
│   │   └── Test_IO.cu
│   ├── beethoven.ply
│   ├── denoised_teapot.pcd
│   ├── Denoise-Op
│   │   ├── Denoise_PtCloud.cu
│   │   ├── Kdtree
│   │   │   ├── kdtree_impl.cpp
│   │   │   ├── kdtree_impl_v2.cpp
│   │   │   ├── kdtree_nn.cpp
│   │   │   ├── Kdtree_NN.txt
│   │   │   ├── kdtree_pcl.cpp
│   │   │   ├── kdtree_search.cpp
│   │   │   ├── kdtree_search_pcl.cpp
│   │   │   └── Makefile
│   │   ├── main.cpp
│   │   └── Makefile
│   ├── Filter-Test
│   │   ├── Filter_Kernel.cu
│   │   ├── main.cpp
│   │   └── Makefile
│   ├── Matrix-Transform
│   │   ├── main.cpp
│   │   ├── Makefile
│   │   └── Matrix_Transform.cu
│   ├── Matrix-Transform-CPP
│   │   ├── main.cpp
│   │   ├── Makefile
│   │   └── Matrix_Transform.cpp
│   ├── scan_Velodyne_VLP16.pcd
│   └── teapot.ply
├── README.md
├── SeparableGaussianFilter-ImageBlur-GPU
│   ├── 138728.jpg
│   ├── 2040735.jpg
│   ├── brad-huchteman-stone-mountain.jpg
│   ├── Image_Blur_Sep_v3.cu
│   ├── index.jpeg
│   ├── Makefile
│   ├── README.md
│   └── UI-Sidewalk-640x480.jpg
├── SeparableGaussianFilter-ImageBlur-Shared-Mem-GPU
│   ├── 138728.jpg
│   ├── 2040735.jpg
│   ├── brad-huchteman-stone-mountain.jpg
│   ├── Image_Blur_Sep_v5.cu
│   ├── index.jpeg
│   ├── Makefile
│   ├── README.md
│   └── UI-Sidewalk-640x480.jpg
└── SeparableMeanFilter-ImageBlur-GPU
    ├── 138728.jpg
    ├── 2040735.jpg
    ├── brad-huchteman-stone-mountain.jpg
    ├── Image_Blur_Sep_v2.cu
    ├── index.jpeg
    ├── Makefile
    ├── README.md
    └── UI-Sidewalk-640x480.jpg

37 directories, 159 files
```

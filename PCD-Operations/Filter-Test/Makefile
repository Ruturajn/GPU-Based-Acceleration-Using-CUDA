#------------------------------------------------------------------------------
# A Makefile for building a Point Cloud Filter CUDA project, which contains
# both .cpp and .cu files. The build files are stored in a separate directory
# called "build".
# 
# Usage: make
# 
# Build Targets:
# 		all      : Builds all the source files and creates an executable.
# 		clean    : Removes all the files created by the build process.
#
# Author : Ruturaj A. Nanoti
# Date   : 13 March, 2022 
#------------------------------------------------------------------------------

# Defining Compiler and related flags
CC = nvcc
CPP_CC = g++
CPP_FLAGS = -g
NVCC_FLAGS = -G -g -w
NVCC_COMPILE_FLAGS = -x cu
ARCH_FLAGS = -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets

LINKER_FLAGS = -lpcl_io -lboost_system -lpcl_common -lpcl_features -lpcl_filters -lpcl_io_ply \
			   -lpcl_kdtree -lpcl_keypoints -lpcl_ml -lpcl_octree -lpcl_outofcore -lpcl_people \
			   -lpcl_recognition -lpcl_registration -lpcl_sample_consensus -lpcl_search \
			   -lpcl_segmentation -lpcl_stereo -lpcl_surface -lpcl_tracking -lpcl_visualization \
			   -lvtkCommonDataModel-7.1 -lvtkCommonCore-7.1 -lvtkRenderingCore-7.1 -lvtkCommonMath-7.1 \


LIBS = -I /usr/local/include/pcl-1.12/ -I /usr/include/eigen3/ -I /usr/include/vtk-7.1/

#------------------------------------------------------------------------------

# Defining the CPP and CUDA source files, also naming the output target
SRC_CPP = main.cpp 
SRC_CU = Filter_Kernel.cu
BUILD_DIR = build

TARGET = app
#------------------------------------------------------------------------------

# Bold
BBlack="\033[0;30m"       # Black
BRed="\033[0;31m"         # Red
BGreen="\033[0;32m"       # Green
BYellow="\033[0;33m"      # Yellow
BBlue="\033[0;34m"        # Blue
BPurple="\033[0;35m"      # Purple
BCyan="\033[0;36m"        # Cyan
BWhite="\033[0;37m"       # White

#------------------------------------------------------------------------------

# Defining pattern rule for building .o files from .cpp files
CPP_OBJS = $(SRC_CPP:.cpp=.o)
%.o : %.cpp
	@echo "\033[1;32m[ * ] Building cpp source file ...\033[0;37m"
	$(CPP_CC) $(CPP_FLAGS) $(LIBS) $(LINKER_FLAGS) -c $< -o ./$(BUILD_DIR)/$@

#------------------------------------------------------------------------------

# Defining pattern rule for building .o files from .cu files
CU_OBJS = $(SRC_CU:.cu=.o)
%.o : %.cu
	@echo "\033[1;33m[ * ] Building cu source file ...\033[0;37m"
	$(CC) $(NVCC_FLAGS) $(NVCC_COMPILE_FLAGS) $(ARCH_FLAGS) $(LIBS) $(LINKER_FLAGS) -dc $< -o ./$(BUILD_DIR)/$@

#------------------------------------------------------------------------------

#Defining the recipes and various targets
.PHONY: all clean

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	@echo "\033[1;35m[ * ] Creating Build Directory ...\033[0;37m"
	mkdir -p $(BUILD_DIR)

$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	@echo "\033[1;34m[ * ] Building the Final Executable ...\033[0;37m"
	$(CC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(LIBS) $(LINKER_FLAGS) ./$(BUILD_DIR)/$(CPP_OBJS) ./$(BUILD_DIR)/$(CU_OBJS) -o ./$(BUILD_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

#------------------------------------------------------------------------------

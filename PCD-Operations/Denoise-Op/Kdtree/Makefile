#------------------------------------------------------------------------------
# A Makefile for building a Point Cloud Denoise project, which 
# contains .cpp files. The build files are stored in a separate 
# directory called "build".
#
# Author : Ruturaj A. Nanoti
# Date   : 1 April, 2022 
#------------------------------------------------------------------------------

# Defining Compiler and related flags
CC = g++
CPP_FLAGS = -g

LINKER_FLAGS = -lpcl_io -lboost_system -lpcl_common -lpcl_features -lpcl_filters -lpcl_io_ply \
			   -lpcl_kdtree -lpcl_keypoints -lpcl_ml -lpcl_octree -lpcl_outofcore -lpcl_people \
			   -lpcl_recognition -lpcl_registration -lpcl_sample_consensus -lpcl_search \
			   -lpcl_segmentation -lpcl_stereo -lpcl_surface -lpcl_tracking -lpcl_visualization \
			   -lvtkCommonDataModel-7.1 -lvtkCommonCore-7.1 -lvtkRenderingCore-7.1 -lvtkCommonMath-7.1


LIBS = -I /usr/local/include/pcl-1.12/ -I /usr/include/eigen3/ -I /usr/include/vtk-7.1/

FILE = one

#------------------------------------------------------------------------------
BUILD_DIR = build
# CPP_TARGET = cpp_app
# CPP_TARGET_2 = cpp_app_v2
# CPP_TARGET_3 = cpp_app_v3
# CPP_TARGET_4 = cpp_app_v4
# CPP_TARGET_5 = cpp_app_v5
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

ifeq ($(FILE), one)
	SRC = kdtree_impl.cpp
	TARGET = kdtree_impl
else ifeq ($(FILE), two)
	SRC = kdtree_impl_v2.cpp
	TARGET = kdtree_impl_v2
else ifeq ($(FILE), three)
	SRC = kdtree_search.cpp
	TARGET = kdtree_search
else ifeq ($(FILE), four)
	SRC = kdtree_nn.cpp
	TARGET = kdtree_nn
else ifeq ($(FILE), five)
	SRC = kdtree_pcl.cpp
	PCL_TARGET = kdtree_pcl
else ifeq ($(FILE), six)
	SRC = kdtree_search_pcl.cpp
	PCL_TARGET = kdtree_search_pcl
endif

#Defining the recipes and various targets
# .PHONY: cppb cppb_v2 cpp_app_v3 cpp_app_v4 clean help
.PHONY : all pcl_target clean help

all : $(BUILD_DIR) $(TARGET)
pcl_target : $(BUILD_DIR) $(PCL_TARGET)

$(BUILD_DIR):
	@echo "\033[1;35m[ * ] Creating Build Directory ...\033[0;37m"
	mkdir -p $(BUILD_DIR)

$(TARGET):
	@echo "\033[1;34m[ * ] Building the Executable ...\033[0;37m"
	$(CC) $(CPP_FLAGS) $(SRC) -o ./$(BUILD_DIR)/$(TARGET)

$(PCL_TARGET):
	@echo "\033[1;34m[ * ] Building the Executable ...\033[0;37m"
	$(CC) $(CPP_FLAGS) $(SRC) $(LIBS) $(LINKER_FLAGS) -o ./$(BUILD_DIR)/$(PCL_TARGET)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "-------------------------------------------------------------------------------------------------"
	@echo "The Following are valid targets for this Makefile :"
	@echo "- cppb    : Builds the executable for kdtree_impl.cpp."
	@echo "- cppb_v2 : Builds the executable for kdtree_impl_v2.cpp."
	@echo "- cppb_v3 : Builds the executable for kdtree_pcl.cpp."
	@echo "- cppb_v4 : Builds the executable for kdtree_search.cpp."
	@echo "- cppb_v5 : Builds the executable for kdtree_search_pcl.cpp."
	@echo "-------------------------------------------------------------------------------------------------"

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# A Makefile for building a Point Cloud Matrix Transform CUDA project, which 
# contains both .cpp and .cu files. The build files are stored in a separate 
# directory called "build".
# 
# Usage: make
# 
# Build Targets:
# 		all      : Builds all the source files and creates an executable.
# 		clean    : Removes all the files created by the build process.
# 		run      : Runs the final executable.
#
# Author : Ruturaj A. Nanoti
# Date   : 14 March, 2022 
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

#------------------------------------------------------------------------------

# Defining the CPP and CUDA source files, also naming the output target
SRC_CPP = main.cpp Matrix_Transform.cpp
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
	$(CC) $(CPP_FLAGS) $(LIBS) $(LINKER_FLAGS) -c $< -o ./$(BUILD_DIR)/$@

#------------------------------------------------------------------------------

#Defining the recipes and various targets
.PHONY: all clean run help

all: $(BUILD_DIR) $(TARGET)

$(BUILD_DIR):
	@echo "\033[1;35m[ * ] Creating Build Directory ...\033[0;37m"
	mkdir -p $(BUILD_DIR)

$(TARGET): $(CPP_OBJS)
	@echo "\033[1;34m[ * ] Building the Final Executable ...\033[0;37m"
	$(CC) $(wildcard ./$(BUILD_DIR)/*.o) -o ./$(BUILD_DIR)/$(TARGET) $(LIBS) $(LINKER_FLAGS) 

run:
	./$(BUILD_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "----------------------------------------------------------------"
	@echo "The Following are valid targets for this Makefile :"
	@echo "- all   : Builds all the source files and creates an executable."
	@echo "- clean : Removes all the files created by the build process."
	@echo "- run   : Runs the final executable."
	@echo "----------------------------------------------------------------"

#------------------------------------------------------------------------------

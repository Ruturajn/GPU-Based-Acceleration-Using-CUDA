CC = nvcc
CPP_CC = g++
NVCC_CFLAGS = -G -g -gencode
ARCH_FLAGS = arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets #-w

LINKER_FLAGS = -lpcl_io -lboost_system -lpcl_common -lpcl_features -lpcl_filters -lpcl_io_ply \
			   -lpcl_kdtree -lpcl_keypoints -lpcl_ml -lpcl_octree -lpcl_outofcore -lpcl_people \
			   -lpcl_recognition -lpcl_registration -lpcl_sample_consensus -lpcl_search \
			   -lpcl_segmentation -lpcl_stereo -lpcl_surface -lpcl_tracking -lpcl_visualization \
			   -lvtkCommonDataModel-7.1 -lvtkCommonCore-7.1 -lvtkRenderingCore-7.1 -lvtkCommonMath-7.1 \


LIBS = -I /usr/local/include/pcl-1.12/ -I /usr/include/eigen3/ -I /usr/include/vtk-7.1/
FILE = one

ifeq ($(FILE), one)
	SRC = Test_IO.cu
else ifeq ($(FILE), two)
	SRC = Filter.cu
endif

.PHONY: all clean cppb cppb1

all: $(FILE).out

$(FILE).out:
	$(CC) $(NVCC_CFLAGS) $(ARCH_FLAGS) $(LIBS) $(LINKER_FLAGS) -o $(FILE).out $(SRC)

cppb:
	$(CPP_CC) -g Test_IO.cpp $(LIBS) $(LINKER_FLAGS) -o Test_IO.out

cppb1:
	$(CPP_CC) -g Matrix_Transform.cpp $(LIBS) $(LINKER_FLAGS) -o Matrix_Transform.out

clean:
	rm -f *.out

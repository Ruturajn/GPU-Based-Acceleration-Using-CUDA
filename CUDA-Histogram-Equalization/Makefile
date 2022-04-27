CC = nvcc
NVCC_FLAGS = -G -g -gencode
ARCH_FLAGS = arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets -rdc=true -lcudadevrt
OPENCV_FLAGS = `pkg-config opencv --cflags --libs`
FILE = one

ifeq ($(FILE), one)
	SRC = Image_Hist_Eql.cu
else ifeq ($(FILE), two)
	SRC = Image_Hist_Eql_8_bit.cu
endif

.PHONY: all clean

all: $(FILE).out

$(FILE).out:
	$(CC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(OPENCV_FLAGS) -o $(FILE).out $(SRC)

clean:
	rm -rf *.out
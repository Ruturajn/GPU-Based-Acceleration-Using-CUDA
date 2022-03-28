CC = nvcc
NVCC_FLAGS = -G -g -gencode
ARCH_FLAGS = arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets
OPENCV_FLAGS = `pkg-config opencv --cflags --libs`
FILE = one
SRC = Vid_Image_Inversion.cu

.PHONY: all clean

all: $(FILE).out

$(FILE).out:
	$(CC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(OPENCV_FLAGS) -o $(FILE).out $(SRC)

clean:
	rm -f *.out

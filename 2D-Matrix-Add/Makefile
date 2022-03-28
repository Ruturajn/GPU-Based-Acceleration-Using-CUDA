CC = nvcc
NVCC_FLAGS = -G -g -gencode
ARCH_FLAGS = arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets
FILE = one
SRC = Add_Matrix.cu

ifeq ($(FILE), two)
	SRC = Add_Matrix_v2.cu
endif

.PHONY: all clean

all: $(FILE).out

$(FILE).out:
	$(CC) $(NVCC_FLAGS) $(ARCH_FLAGS) -o $(FILE).out $(SRC)

clean:
	rm -f *.out

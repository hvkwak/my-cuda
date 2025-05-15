##
# my cuda-programming-works
#
# @file
# @version 0.2
#
# notes
# $@ = the target, on the left of the colon. e.g. main
# $^ = All prerequisites
# $< = the first prerequisite
#
# run with e.g. make EXERCISE=exercise2

# Default build exercise1
EXERCISE ?= exercise1

# Compiler and flags
NVCC = nvcc

## exercise 1
# exercise 1-4: -arch sm_20 was for Fermi architecture. If this is not specified, it sets to sm_52 default value.
ifeq ($(EXERCISE),exercise1)

TARGET = main
NVCC_FLAGS = -O2
SRC = $(EXERCISE)/*.cu

all: $(TARGET)
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
clean:
	rm -f $(TARGET)
endif


## exercise 2
ifeq ($(EXERCISE),exercise2)
CU_APPS=checkDeviceInfor checkThreadIndex sumArraysOnGPU-timer \
        sumMatrixOnGPU-1D-grid-1D-block sumMatrixOnGPU-2D-grid-2D-block \
        checkDimension defineGridBlock sumArraysOnGPU-small-case \
        sumMatrixOnGPU-2D-grid-1D-block sumMatrixOnGPU \
	sumMatrixOnGPU-2D-grid-2D-block-integer
C_APPS=sumArraysOnHost
NVCC_FLAGS = -O2
all: ${C_APPS} ${CU_APPS}

%: $(EXERCISE)/%.cu
	nvcc ${NVCC_FLAGS} -o $@ $<
%: $(EXERCISE)/%.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
endif


## exercise3
## sm_52 statt sm_35?
##
## keep it for later use:
## more files
## nestedHelloWorld nestedReduce nestedReduce2 nestedReduceNosync \
		reduceInteger simpleDeviceQuery simpleDivergence sumMatrix
## nvcc -O2 -arch=sm_52 -o $@ $< -lcudadevrt --relocatable-device-code true
ifeq ($(EXERCISE),exercise3)
CU_APPS=nestedReduce reduceInteger

C_APPS=
NVCC_FLAGS = -O2
all: ${C_APPS} ${CU_APPS}
%: $(EXERCISE)/%.cu
	nvcc -O2 -arch=sm_52 -o $@ $< -lcudadevrt --relocatable-device-code true
%: $(EXERCISE)/%.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
endif
# end

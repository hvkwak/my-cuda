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



## exercise 1
# exercise 1-4: -arch sm_20 what's for Fermi architecture. If this is not specified, it sets to sm_52 default value.
ifeq ($(EXERCISE),exercise1)
NVCC_FLAGS = -O2
SRC = $(EXERCISE)/*.cu
all: main
$(TARGET): $(SRC)
	nvcc $(NVCC_FLAGS) -o $@ $^
clean:
	rm -f main
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
# nestedHelloWorld nestedReduce nestedReduce2 nestedReduceNosync \
		reduceInteger simpleDeviceQuery simpleDivergence sumMatrix
# use to enable recursive functions nvcc -O2 -arch=sm_52 -o $@ $< -lcudadevrt --relocatable-device-code true
ifeq ($(EXERCISE),exercise3)
CU_APPS=nestedReduce reduceInteger reduceFloat nestedHelloWorld
C_APPS=
NVCC_FLAGS = -O2 -lcudadevrt --relocatable-device-code true
all: ${C_APPS} ${CU_APPS}
%: $(EXERCISE)/%.cu
	nvcc ${NVCC_FLAGS} -arch=sm_75 -o $@ $<
%: $(EXERCISE)/%.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
endif


## exercise4
ifeq ($(EXERCISE), exercise4)
CU_APPS=globalVariable memTransfer pinMemTransfer readSegment \
		readSegmentUnroll simpleMathAoS simpleMathSoA sumArrayZerocpy \
		sumMatrixGPUManaged sumMatrixGPUManual transpose writeSegment \
		sumArrayZerocpyUVA
C_APPS=

all: ${C_APPS} ${CU_APPS}

sumArrayZerocpyL1CacheDisabled: $(EXERCISE)/sumArrayZerocpy.cu
	nvcc -O2 -arch=sm_20 -Xptxas -dlcm=cg -o $@ $<

%: $(EXERCISE)/%.cu
	nvcc -O2 -arch=sm_75 -o $@ $<

%: $(EXERCISE)/%.c
	gcc -O2 -std=c99 -o $@ $<

clean:
	rm -f ${CU_APPS} ${C_APPS}

endif

##
# my cuda-programming-works
#
# @file
# @version 0.2
#
#
# run with e.g. make EXERCISE=exercise2

# Default build exercise1
EXERCISE ?= exercise1

# Output binary name
TARGET = main

# choose source files
SRC = $(EXERCISE)/*.cu

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O2

# Dynamically set flags based on EXERCISE
# note that NVCC_FLAGS = -O2 -D$(EXERCISE) can be an option, too..
#ifeq ($(EXERCISE),exercise1)
#    NVCC_FLAGS = -O2 -DEXERCISE1
#endif
#ifeq ($(EXERCISE),exercise2)
#    NVCC_FLAGS = -O2 -DEXERCISE2
#endif

# Default target
# $@ = main (the target on the left of the colon)
# $^ = All prerequisites
# erst ohne .o files

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
# end

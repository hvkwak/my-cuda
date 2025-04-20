##
# my cuda-programming-works
#
# @file
# @version 0.1
# Name of the final executable
TARGET = main

# Source files
SRC = main.cu

# Compiler
NVCC = nvcc

# Compilation flags
NVCC_FLAGS = -O2

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
# end

CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I./
LIBS         := -lcusparse

all: spmm_blockedell_example

spmm_blockedell_example: spmm_blockedell_example.cpp
	nvcc $(INC) spmm_blockedell_example.cpp mmio.c smsh.c -o spmm_blockedell_example $(LIBS)
	gcc mmio.h 

clean:
	rm -f spmm_blockedell_example

test:
	@echo "\n==== SpMM BLOCKED ELL Test ====\n"
	./spmm_blockedell_example

.PHONY: clean all test

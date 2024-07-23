CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include -I./
LIBS         := -lcusparse

all: smart_1 smart_2 smart_3

smart_1: smart_1.cpp
	nvcc -std=c++11 $(INC) smart_1.cpp mmio.c smsh.c -o smart_1 $(LIBS)
	gcc mmio.h 
	
smart_2: smart_2.cpp
	nvcc -std=c++11 $(INC) smart_2.cpp mmio.c smsh.c -o smart_2 $(LIBS)
	gcc mmio.h 
	
smart_3: smart_3.cpp
	nvcc -std=c++11 $(INC) smart_3.cpp mmio.c smsh.c -o smart_3 $(LIBS)
	gcc mmio.h 

clean:
	rm -f smart_1 smart_2 smart_3

.PHONY: clean all test

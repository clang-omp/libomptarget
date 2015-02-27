##===- RTLs/cuda/Makefile ----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build an RTL for a cuda machine if available
#
##===----------------------------------------------------------------------===##

#locate cuda libraries

cudalibs  := 

ifneq ($(wildcard /usr/local/cuda/lib64),)
cudalibs += -L/usr/local/cuda/lib64
endif
ifneq ($(wildcard /usr/local/cuda/lib),)
cudalibs += -L/usr/local/cuda/lib
endif

cudaincludes := 
ifneq ($(wildcard /usr/local/cuda/include),)
cudaincludes += -I/usr/local/cuda/include
endif

cudalib := $(shell ldconfig -p | grep 'libcuda.so ' | grep -o '/.*' ) 

build_cuda :=
ifneq ($(cudalib),)
ifneq ($(cudalibs),)
ifneq ($(cudaincludes),)
	build_cuda := true
endif
endif
endif

CPP_FILES = $(wildcard src/*.cpp)
INC_FILES = $(wildcard src/*.h)
OBJ_FILES = $(subst src/,obj/,$(CPP_FILES:.cpp=.o))

CC := g++
CFLAGS := -c -fPIC -I src/ -I../../src/ $(cudaincludes) -DTARGET_NAME=CUDA
LDFLAGS := -shared -lelf -lcuda -lcudart $(cudalibs)

ifdef OMPTARGET_DEBUG
CFLAGS += -g -DOMPTARGET_DEBUG -DCUDA_ERROR_REPORT
endif

ifneq ($(build_cuda),)
../../lib/libomptarget.rtl.cuda.so : $(OBJ_FILES)
	@ echo "Building CUDA target RTL" && \
	mkdir -p ../../lib
	$(CC) $(LDFLAGS) -o $@ $(OBJ_FILES)

else
all : 
	@ echo "CUDA not found in system: not building CUDA RTL"

endif

obj/%.o: src/%.cpp $(INC_FILES)
	@ mkdir -p obj
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf obj
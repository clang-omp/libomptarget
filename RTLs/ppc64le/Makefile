##===- RTLs/ppc64/Makefile ----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build an RTL for a ppc64le machine if available
#
##===----------------------------------------------------------------------===##

#identify machine we are building
#machine := $(lastword $(subst ., ,$(shell uname -r)))
machine := $(shell uname -m)

tmachine := ppc64le
tmachine_name := PPC64le
tmachine_libname := ppc64

CPP_FILES = $(wildcard ../generic-64bit/src/*.cpp)
INC_FILES = $(wildcard ../generic-64bit/src/*.h)
OBJ_FILES = $(subst ../generic-64bit/src/,obj/,$(CPP_FILES:.cpp=.o))

CC := g++
CFLAGS := -c -fPIC -I src/ -I../../src/ -DTARGET_NAME=$(tmachine_name)
LDFLAGS := -shared -lelf -lffi

ifdef OMPTARGET_DEBUG
CFLAGS += -g -DOMPTARGET_DEBUG
endif

ifeq ($(machine),$(tmachine))
../../lib/libomptarget.rtl.$(tmachine_libname).so : $(OBJ_FILES)
	@ echo "Building $(tmachine_name) target RTL" && \
	mkdir -p ../../lib
	$(CC) $(LDFLAGS) -o $@ $(OBJ_FILES)
	
else
all : 
	@ echo "$(tmachine_name) not found in system: not building $(tmachine_name) RTL"
	
endif

obj/%.o: ../generic-64bit/src/%.cpp $(INC_FILES)
	@ mkdir -p obj
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf obj

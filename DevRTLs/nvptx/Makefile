##===- DevRTLs/nvptx/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build the NVPTX (CUDA) Device RTL if the CUDA tools are available
#
##===----------------------------------------------------------------------===##

CUD_FILES := $(wildcard src/*.cu)
INC_FILES := $(wildcard src/*.h)
OBJ_FILES := $(subst src/,obj/,$(CUD_FILES:.cu=.o))

NVCC := nvcc

ifneq ($(OMPTARGET_NVPTX_SM),)
  _comma:= ,
  _empty:=
  _space:= $(_empty) $(empty)
  CFLAGS += $(foreach sm,$(subst $(_comma),$(_space),$(OMPTARGET_NVPTX_SM)), \
  		-gencode arch=compute_$(sm),code=sm_$(sm))
else
  CFLAGS += -arch sm_35
endif  
CFLAGS += -dc

LDFLAGS := -lcuda --lib

build_cuda := $(shell which $(NVCC))

ifdef OMPTARGET_NVPTX_DEBUG
  CFLAGS += -DOMPTARGET_NVPTX_DEBUG=-1 -g --ptxas-options=-v
endif

.PHONY: clean

ifneq ($(build_cuda),)
all : ../../lib/libomptarget-nvptx.a
	@ echo "Built NVPTX device RTL" 

else
all : 
	@ echo "CUDA tools $$build_cuda not found in system: not building NVPTX device RTL"

endif

../../lib/libomptarget-nvptx.a : $(OBJ_FILES)
	@ mkdir -p ../../lib
	$(NVCC) $(LDFLAGS) -o $@ $(OBJ_FILES)

obj/%.o: src/%.cu $(INC_FILES)
	@ mkdir -p obj
	$(NVCC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf obj lib
	rm -f src/*~

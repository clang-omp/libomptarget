##===- RTLs/Makefile ---------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build an RTL for this machine is available
#
##===----------------------------------------------------------------------===##

#RTL_FOLDERS = $(wildcard */)
RTL_FOLDERS = cuda ppc64 ppc64le x86_64 


all:
	@for f in $(RTL_FOLDERS); do \
		make -C $$f ; \
	done 
	
clean:
	@for f in $(RTL_FOLDERS); do \
		make -C $$f clean ; \
	done 

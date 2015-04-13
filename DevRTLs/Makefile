##===- RTLs/Makefile ---------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build the device RTL for each available machine available
#
##===----------------------------------------------------------------------===##

DEV_RTL_FOLDERS = nvptx 

all:
	@for f in $(DEV_RTL_FOLDERS); do \
		make -C $$f -j16 ; \
	done 
	
clean:
	@for f in $(DEV_RTL_FOLDERS); do \
		make -C $$f clean ; \
	done 

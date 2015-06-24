##===----------------------------------------------------------------------===##
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

if(CUDA_FOUND)

    message("Building CUDA target RTL")

    add_definitions(-DTARGET_NAME=CUDA)
	
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        add_definitions(-DCUDA_ERROR_REPORT)
    endif()

    add_library(omptarget.rtl.cuda SHARED src/rtl.cpp)
    target_link_libraries(omptarget.rtl.cuda ${CUDA_LIBRARIES} cuda ${LIBELF_LIBRARIES})

else()

    message("CUDA not found in system: not building CUDA RTL")

endif()

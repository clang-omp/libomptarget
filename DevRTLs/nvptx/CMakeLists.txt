##===----------------------------------------------------------------------===##
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

if(CUDA_FOUND)

    set(cuda_src_files
        src/cancel.cu
        src/critical.cu
        src/debug.cu
        src/libcall.cu
        src/loop.cu
        src/omptarget-nvptx.cu
        src/parallel.cu
        src/reduction.cu
        src/stdio.cu
        src/sync.cu
        src/task.cu
    )

    include_directories(
        src/
    )

    message("Building NVPTX device RTL")

    if(DEFINED OMPTARGET_NVPTX_SM)
        string(REPLACE "," ";" omptarget_nvptx_sm_list ${OMPTARGET_NVPTX_SM})
        foreach(sm ${omptarget_nvptx_sm_list})
           set(CUDA_ARCH ${CUDA_ARCH} -gencode arch=compute_${sm},code=sm_${sm})
        endforeach()
    else()
        set(CUDA_ARCH -arch sm_35)
    endif()
  
    set(BUILD_SHARED_LIBS OFF)
    set(CUDA_SEPARABLE_COMPILATION ON)

    cuda_add_library(omptarget-nvptx STATIC ${cuda_src_files} 
        OPTIONS ${CUDA_ARCH})

    target_link_libraries(omptarget-nvptx ${CUDA_LIBRARIES})

else()

    message("CUDA tools not found in system: not building NVPTX device RTL")

endif()

##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build a preleminary version of libomptarget.so
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(libomptarget)

# Configure ouput folder for generated libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Configure path of custom cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/)

find_package(LibElf REQUIRED)
if(LIBELF_FOUND)
    include_directories(${LIBELF_INCLUDE_DIRS})
endif()

find_package(CUDA QUIET)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Allow support for C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DOMPTARGET_DEBUG)
endif()

set(src_files
    src/omptarget.cpp
    src/targets_info.cpp
)

include_directories(src/)

# Build omptarget library
add_library(omptarget SHARED ${src_files})
target_link_libraries(omptarget ${LIBELF_LIBRARIES} dl)

# Build RTL libraries when available
add_subdirectory(RTLs)
add_subdirectory(DevRTLs)


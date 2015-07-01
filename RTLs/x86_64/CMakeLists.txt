##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
# Build an RTL for a x86_64 machine if available
#
##===----------------------------------------------------------------------===##

set(tmachine "x86_64")
set(tmachine_name "x86_64")
set(tmachine_libname "x86_64")

if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES ${tmachine}$)

    find_package(LibFFI QUIET)

    if(LIBFFI_FOUND)

        message("Building " ${tmachine_name} " target RTL")

        include_directories(${LIBFFI_INCLUDE_DIR})

        add_definitions(-DTARGET_NAME=${tmachine_name})

        add_library(omptarget.rtl.${tmachine_libname} SHARED ../generic-64bit/src/rtl.cpp)
        target_link_libraries(omptarget.rtl.${tmachine_libname}
            ${LIBFFI_LIBRARIES} ${LIBELF_LIBRARIES}
        )

    else()

        message("LibFFI not found in system: not building " ${tmachine_name} " RTL")

    endif()

else()

    message(${tmachine_name} " not found in system: not building " ${tmachine_name} " RTL")

endif()

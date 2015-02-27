//===-------- targets_info.cpp - Information about Target RTLs ---- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Map between ELF machine IDs and the RTL library that supports it
//
//===----------------------------------------------------------------------===//


#include "targets_info.h"

static targets_info_table_entry targets_info_entries[] = {
    {   21 /* EM_PPC64 */ ,   "libomptarget.rtl.ppc64.so"},
    {   62 /* EM_X86_64*/ ,   "libomptarget.rtl.x86_64.so"},
    {  190 /* EM_CUDA */  ,   "libomptarget.rtl.cuda.so"}
};

targets_info_table targets_info = {
    sizeof(targets_info_entries) / sizeof(targets_info_table_entry),
    &targets_info_entries[0]
};


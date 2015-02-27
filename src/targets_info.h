//===-------- targets_info.h - Information about Target RTLs ------ C++ -*-===//
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

#include <stdint.h>

#ifndef _TARGETS_INFO_H_
#define _TARGETS_INFO_H_

struct targets_info_table_entry{
  uint16_t Machine_Elf_ID;
  const char *Machine_RTL_Lib;
};
struct targets_info_table{
  int32_t Number_of_Entries;
  targets_info_table_entry *Entries;
};

#endif

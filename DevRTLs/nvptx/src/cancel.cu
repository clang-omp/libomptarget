//===------ cancel.cu - NVPTX OpenMP cancel interface ------------ CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to be used in the implementation of OpenMP cancel.
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

EXTERN int32_t __kmpc_cancellationpoint(
  kmp_Indent* loc, 
  int32_t global_tid, 
  int32_t cancelVal)
{
  PRINT(LD_IO, "call kmpc_cancellationpoint(cancel val %d)\n", cancelVal);
  // disabled
  return FALSE;
}

EXTERN int32_t __kmpc_cancel(
  kmp_Indent* loc, 
  int32_t global_tid, 
  int32_t cancelVal)
{
  PRINT(LD_IO, "call kmpc_cancel(cancel val %d)\n", cancelVal);
  // disabled
  return FALSE;
}

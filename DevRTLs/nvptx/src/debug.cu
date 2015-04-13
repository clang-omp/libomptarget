//===------------ debug.cu - NVPTX OpenMP debug utilities -------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of debug utilities to be
// used in the application.
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

////////////////////////////////////////////////////////////////////////////////
// print current state
////////////////////////////////////////////////////////////////////////////////

NOINLINE void PrintTaskDescr(omptarget_nvptx_TaskDescr *taskDescr, char *title, int level)
{
  omp_sched_t sched = taskDescr->GetRuntimeSched();
  PRINT(LD_ALL, "task descr %s %d: %s, in par %d, dyn %d, rt sched %d, chunk %lld;"
    "  tid %d, tnum %d, nthreads %d\n",
    title, level, (taskDescr->IsParallelConstruct()?"par":"task"), 
    taskDescr->InParallelRegion(), taskDescr->IsDynamic(), 
    sched, taskDescr->RuntimeChunkSize(),
    taskDescr->ThreadId(), taskDescr->ThreadsInTeam(), taskDescr->NThreads());
}

////////////////////////////////////////////////////////////////////////////////
// debug for compiler (should eventually all vanish)
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_print_str(char *title)
{
  PRINT(LD_ALL, " %s\n", title);
}

EXTERN void __kmpc_print_title_int(char *title, int data)
{
  PRINT(LD_ALL, "%s val=%d\n", title, data);
}

EXTERN void __kmpc_print_index(char *title, int i)
{
  PRINT(LD_ALL, "i = %d\n", i);
}

EXTERN void __kmpc_print_int(int data)
{
  PRINT(LD_ALL, "val=%d\n", data);
}

EXTERN void __kmpc_print_double(double data)
{
  PRINT(LD_ALL, "val=%lf\n", data);
}

EXTERN void __kmpc_print_address_int64(int64_t data)
{
  PRINT(LD_ALL, "val=%016llx\n", data);
}

////////////////////////////////////////////////////////////////////////////////
// substitute for printf in kernel (should vanish)
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_kernel_print(char *title) 
{
  PRINT(LD_ALL, " %s\n", title);
}

EXTERN void __kmpc_kernel_print_int8(char *title, int64_t data) 
{
  PRINT(LD_ALL, " %s val=%lld\n", title, data);
}

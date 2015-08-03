//===--- omptarget-nvptx.cu - NVPTX OpenMP GPU initialization ---- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the initialization code for the GPU
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

__device__ omptarget_nvptx_TeamDescr omptarget_nvptx_teamContexts[MAX_NUM_TEAMS];
__device__ omptarget_nvptx_ThreadPrivateContext omptarget_nvptx_threadPrivateContext;
__device__ omptarget_nvptx_GlobalICV omptarget_nvptx_globalICV;

////////////////////////////////////////////////////////////////////////////////
// init entry points
////////////////////////////////////////////////////////////////////////////////


EXTERN void __kmpc_kernel_init(int ThreadLimit)
{
  PRINT(LD_IO, "call to __kmpc_kernel_init with version %f\n", OMPTARGET_NVPTX_VERSION);
  // init thread private
  int globalThreadId = GetGlobalThreadId();
  omptarget_nvptx_threadPrivateContext.InitThreadPrivateContext(globalThreadId);
  
  int threadIdInBlock = GetThreadIdInBlock();
  if (threadIdInBlock == TEAM_MASTER) {
    PRINT0(LD_IO, "call to __kmpc_kernel_init for master\n");
    // init global icv
    omptarget_nvptx_globalICV.gpuCycleTime = 1.0 / 745000000.0; // host reports 745 mHz
    omptarget_nvptx_globalICV.cancelPolicy = FALSE;  // currently false only
    // init team context
    omptarget_nvptx_TeamDescr & currTeamDescr = getMyTeamDescriptor();
    currTeamDescr.InitTeamDescr();
    // this thread will start execution... has to update its task ICV
    // to points to the level zero task ICV. That ICV was init in
    // InitTeamDescr()
    omptarget_nvptx_threadPrivateContext.SetTopLevelTaskDescr(globalThreadId, 
      currTeamDescr.LevelZeroTaskDescr());
    
    // set number of threads and thread limit in team to started value
    int globalThreadId = GetGlobalThreadId();
    omptarget_nvptx_TaskDescr *currTaskDescr =
        omptarget_nvptx_threadPrivateContext.GetTopLevelTaskDescr(
            globalThreadId);
    currTaskDescr->NThreads() = GetNumberOfThreadsInBlock();
    currTaskDescr->ThreadLimit() = ThreadLimit;
  }
}


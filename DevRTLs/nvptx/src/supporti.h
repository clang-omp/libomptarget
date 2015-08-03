//===--------- supporti.h - NVPTX OpenMP support functions ------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Wrapper implementation to some functions natively supported by the GPU.
//
//===----------------------------------------------------------------------===//

////////////////////////////////////////////////////////////////////////////////
// support: get info from machine
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// machine: get number of (assuming 1D layout)

INLINE int GetNumberOfThreadsInBlock()
{
  return blockDim.x;
}

INLINE int GetNumberOfWarpsInBlock()
{
  ASSERT(LT_FUSSY, GetNumberOfThreadsInBlock() % warpSize == 0, 
    "expected threads num %d to be a multiple of warp size %d\n", 
    GetNumberOfThreadsInBlock(), warpSize);
   return GetNumberOfThreadsInBlock() / warpSize;
}

INLINE int GetNumberOfBlocksInKernel()
{
  return gridDim.x;
}


////////////////////////////////////////////////////////////////////////////////
// machine: get ids  (assuming 1D layout) 

INLINE int GetThreadIdInBlock()
{
  return threadIdx.x;
}

INLINE int GetWarpIdInBlock()
{
  ASSERT(LT_FUSSY, GetNumberOfThreadsInBlock() % warpSize == 0, 
    "expected threads num %d to be a multiple of warp size %d\n", 
    GetNumberOfThreadsInBlock(), warpSize);
  return  GetThreadIdInBlock() / warpSize;
}

INLINE int GetBlockIdInKernel()
{
  return blockIdx.x;
}


////////////////////////////////////////////////////////////////////////////////
// Global thread id used to locate thread info 

INLINE int GetGlobalThreadId()
{
  #ifdef OMPTHREAD_IS_WARP
    return GetBlockIdInKernel() * GetNumberOfWarpsInBlock()   + GetWarpIdInBlock();
  #else
    return GetBlockIdInKernel() * GetNumberOfThreadsInBlock() + GetThreadIdInBlock();
  #endif
}

INLINE int GetNumberOfGlobalThreadIds() 
{
  #ifdef OMPTHREAD_IS_WARP
    return GetNumberOfWarpsInBlock()   * GetNumberOfBlockInKernel();
  #else
    return GetNumberOfThreadsInBlock() * GetNumberOfBlocksInKernel();
  #endif
}

////////////////////////////////////////////////////////////////////////////////
// global  team id used to locate team info 

INLINE int GetGlobalTeamId()
{
  return GetBlockIdInKernel();
}
 
INLINE int GetNumberOfGlobalTeamIds()
{
  return GetNumberOfBlocksInKernel();
}   

////////////////////////////////////////////////////////////////////////////////
// OpenMP Thread id linked to OpenMP

INLINE int GetOmpThreadId(int globalThreadId)    
{
  // omp_thread_num
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext.GetTopLevelTaskDescr(globalThreadId);
  int rc = currTaskDescr->ThreadId();
  return rc;
}

INLINE int GetNumberOfOmpThreads(int globalThreadId)
{
  // omp_num_threads
  omptarget_nvptx_TaskDescr *currTaskDescr =
      omptarget_nvptx_threadPrivateContext.GetTopLevelTaskDescr(globalThreadId);

  ASSERT0(LT_FUSSY, currTaskDescr, "expected a top task descr");

  int rc = currTaskDescr->ThreadsInTeam();
  return rc;
}


////////////////////////////////////////////////////////////////////////////////
// Team id linked to OpenMP

INLINE int GetOmpTeamId()
{
  // omp_team_num
  return GetGlobalTeamId(); // assume 1 block per team
}

INLINE int GetNumberOfOmpTeams()
{
  // omp_num_teams
  return GetNumberOfGlobalTeamIds(); // assume 1 block per team
}


////////////////////////////////////////////////////////////////////////////////
// get OpenMP number of procs

INLINE int GetNumberOfProcsInTeam()
{
  #ifdef OMPTHREAD_IS_WARP
    return GetNumberOfWarpsInBlock();
  #else
    return GetNumberOfThreadsInBlock();
  #endif
}


////////////////////////////////////////////////////////////////////////////////
// Masters

INLINE int IsTeamMaster(int ompThreadId)
{
  return (ompThreadId == 0);
}

INLINE int IsWarpMaster(int ompThreadId)
{
  return (ompThreadId % warpSize == 0);
}


////////////////////////////////////////////////////////////////////////////////
// Memory 
////////////////////////////////////////////////////////////////////////////////

INLINE unsigned long PadBytes(
  unsigned long size, 
  unsigned long alignment) // must be a power of 2
{
  // compute the necessary padding to satify alignment constraint
  ASSERT(LT_FUSSY, (alignment & (alignment - 1)) == 0, 
    "alignment %ld is not a power of 2\n", alignment);
  return (~(unsigned long) size + 1) & (alignment - 1);
}

INLINE void *SafeMalloc(size_t size, const char *msg) // check if success
{
  void * ptr = malloc(size);
  PRINT(LD_MEM, "malloc data of size %d for %s: 0x%llx\n", size, msg,
    P64(ptr)); 
  ASSERT(LT_SAFETY, ptr, "failed to allocate %d bytes for %s\n", size, msg);
  return ptr;
}

INLINE void *SafeFree(void *ptr, const char *msg)
{
  PRINT(LD_MEM, "free data ptr 0x%llx for %s\n", P64(ptr), msg); 
  free(ptr);
  return NULL;
}

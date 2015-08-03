//===---- omptarget-nvptx.h - NVPTX OpenMP GPU initialization ---- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of all library macros, types,
// and functions.
//
//===----------------------------------------------------------------------===//

#ifndef __OMPTARGET_NVPTX_H
#define __OMPTARGET_NVPTX_H

// std includes
#include <stdlib.h>
#include <stdint.h>

// cuda includes
#include <cuda.h>
#include <math.h>

// local includes
#include "option.h"         // choices we have
#include "interface.h"      // interfaces with omp, compiler, and user
#include "debug.h"          // debug

#include "support.h"
#include "counter_group.h"

#define OMPTARGET_NVPTX_VERSION 1.1

// used by the library for the interface with the app
#define DISPATCH_FINISHED 0
#define DISPATCH_NOTFINISHED 1

// used by dynamic scheduling
#define FINISHED 0
#define NOT_FINISHED 1
#define LAST_CHUNK 2


#define TEAM_MASTER 0
#define BARRIER_COUNTER 0
#define ORDERED_COUNTER 1

////////////////////////////////////////////////////////////////////////////////
// global ICV

typedef struct omptarget_nvptx_GlobalICV {
  double  gpuCycleTime; // currently statically determined, should be set by host
  uint8_t cancelPolicy; // 1 bit: enabled (true) or disabled (false)
} omptarget_nvptx_GlobalICV;

////////////////////////////////////////////////////////////////////////////////
// task ICV and (implicit & explicit) task state

class omptarget_nvptx_TaskDescr {
 public:
  // methods for flags
  INLINE omp_sched_t GetRuntimeSched();
  INLINE void SetRuntimeSched(omp_sched_t sched);
  INLINE int  IsDynamic() { return data.items.flags & TaskDescr_IsDynamic; }
  INLINE void SetDynamic() { data.items.flags = data.items.flags | TaskDescr_IsDynamic; }
  INLINE void ClearDynamic() { data.items.flags = data.items.flags & (~TaskDescr_IsDynamic); }
  INLINE int  InParallelRegion() { return data.items.flags & TaskDescr_InPar; }
  INLINE int  IsParallelConstruct() { return data.items.flags & TaskDescr_IsParConstr; }
  INLINE int  IsTaskConstruct() { return ! IsParallelConstruct(); }
  // methods for other fields
  INLINE uint16_t & NThreads() { return data.items.nthreads; }
  INLINE uint16_t & ThreadLimit() { return data.items.threadlimit; }
  INLINE uint16_t & ThreadId() { return data.items.threadId; }
  INLINE uint16_t & ThreadsInTeam() { return data.items.threadsInTeam; }
  INLINE uint64_t & RuntimeChunkSize() { return data.items.runtimeChunkSize; }
  INLINE omptarget_nvptx_TaskDescr * GetPrevTaskDescr() { return prev; }
  INLINE void SetPrevTaskDescr(omptarget_nvptx_TaskDescr *taskDescr) { prev = taskDescr; }
  // init & copy
  INLINE void InitLevelZeroTaskDescr();
  INLINE void Copy(omptarget_nvptx_TaskDescr *sourceTaskDescr);
  INLINE void CopyData(omptarget_nvptx_TaskDescr *sourceTaskDescr);
  INLINE void CopyParent(omptarget_nvptx_TaskDescr *parentTaskDescr);
  INLINE void CopyForExplicitTask(omptarget_nvptx_TaskDescr *parentTaskDescr);
  INLINE void CopyToWorkDescr(omptarget_nvptx_TaskDescr *masterTaskDescr, uint16_t tnum);
  INLINE void CopyFromWorkDescr(omptarget_nvptx_TaskDescr *workTaskDescr);

 private:
  /* bits for flags: (6 used, 2 free)
      3 bits (SchedMask) for runtime schedule 
      1 bit (IsDynamic) for dynamic schedule (false = static)
      1 bit (InPar) if this thread has encountered one or more parallel region
      1 bit (IsParConstr) if ICV for a parallel region (false = explicit task)
   */
  static const uint8_t TaskDescr_SchedMask   = (0x1 | 0x2 | 0x4); 
  static const uint8_t TaskDescr_IsDynamic   = 0x8;  
  static const uint8_t TaskDescr_InPar       = 0x10; 
  static const uint8_t TaskDescr_IsParConstr = 0x20; 

  union { // both have same size
    uint64_t vect[2];
    struct TaskDescr_items {
      uint8_t  flags; // 6 bit used (see flag above)
      uint8_t  unused; 
      uint16_t nthreads; // thread num for subsequent parallel regions
      uint16_t threadlimit; // thread limit ICV
      uint16_t threadId; // thread id
      uint16_t threadsInTeam; // threads in current team
      uint64_t runtimeChunkSize; // runtime chunk size
    } items;
  } data;
  omptarget_nvptx_TaskDescr *prev;
};

// build on kmp
typedef struct omptarget_nvptx_ExplicitTaskDescr {
  omptarget_nvptx_TaskDescr taskDescr; // omptarget_nvptx task description (must be first)
  kmp_TaskDescr   kmpTaskDescr; // kmp task description (must be last)
} omptarget_nvptx_ExplicitTaskDescr;

////////////////////////////////////////////////////////////////////////////////
// thread private data (struct of arrays for better coalescing)
// tid refers here to the global thread id
// do not support multiple concurrent kernel a this time
class omptarget_nvptx_ThreadPrivateContext {
public:
  // task
  INLINE omptarget_nvptx_TaskDescr *Level1TaskDescr(int gtid) { return & levelOneTaskDescr[gtid]; }
  INLINE void SetTopLevelTaskDescr(int gtid, omptarget_nvptx_TaskDescr *taskICV) { topTaskDescr[gtid] = taskICV; }
  INLINE omptarget_nvptx_TaskDescr *GetTopLevelTaskDescr(int gtid);
  // parallel
  INLINE uint16_t & NumThreadsForNextParallel(int gtid) { return tnumForNextPar[gtid]; }
  // sync
  INLINE Counter & Priv(int gtid)               { return priv[gtid]; }
  INLINE void IncrementPriv(int gtid, Counter val) { priv[gtid] += val; }
  // schedule (for dispatch)
  INLINE kmp_sched_t & ScheduleType(int gtid)   { return schedule[gtid]; }
  INLINE int64_t     & Chunk(int gtid)          { return chunk[gtid]; }
  INLINE int64_t     & LoopUpperBound(int gtid) { return loopUpperBound[gtid]; }
  // state for dispatch with dyn/guided
  INLINE Counter & CurrentEvent(int gtid)  { return currEvent_or_nextLowerBound[gtid]; }
  INLINE Counter & EventsNumber(int gtid)  { return eventsNum_or_stride[gtid]; }
  // state for dispatch with static
  INLINE Counter & NextLowerBound(int gtid) { return currEvent_or_nextLowerBound[gtid]; }
  INLINE Counter & Stride(int gtid)         { return eventsNum_or_stride[gtid]; }


  INLINE void InitThreadPrivateContext(int gtid);

private:
  // task ICV for implict threads in the only parallel region
  omptarget_nvptx_TaskDescr levelOneTaskDescr[MAX_NUM_OMP_THREADS];
  // pointer where to find the current task ICV (top of the stack) 
  omptarget_nvptx_TaskDescr *topTaskDescr[MAX_NUM_OMP_THREADS];
  // parallel
  uint16_t tnumForNextPar[MAX_NUM_OMP_THREADS];
  // sync
  Counter priv[MAX_NUM_OMP_THREADS];
  // schedule (for dispatch)
  kmp_sched_t schedule[MAX_NUM_OMP_THREADS]; // remember schedule type for #for
  int64_t chunk[MAX_NUM_OMP_THREADS];
  int64_t loopUpperBound[MAX_NUM_OMP_THREADS];
  // state for dispatch with dyn/guided OR static (never use both at a time)
  Counter currEvent_or_nextLowerBound[MAX_NUM_OMP_THREADS];
  Counter eventsNum_or_stride[MAX_NUM_OMP_THREADS];
};

////////////////////////////////////////////////////////////////////////////////
// Descriptor of a parallel region (worksharing in general)

class omptarget_nvptx_WorkDescr {

 public:
  // access to data
  INLINE omptarget_nvptx_CounterGroup & CounterGroup() { return cg; }
  INLINE omptarget_nvptx_TaskDescr * WorkTaskDescr() { return & masterTaskICV; }
  // init
  INLINE void InitWorkDescr();

private:
  omptarget_nvptx_CounterGroup cg; // for barrier (no other needed)
  omptarget_nvptx_TaskDescr masterTaskICV;
  bool hasCancel;
};


////////////////////////////////////////////////////////////////////////////////
// thread private data (struct of arrays for better coalescing)

class omptarget_nvptx_TeamDescr {
 public:
  // access to data
  INLINE omptarget_nvptx_TaskDescr *LevelZeroTaskDescr() { return &levelZeroTaskDescr; }
  INLINE omptarget_nvptx_WorkDescr & WorkDescr() { return workDescrForActiveParallel; }
  INLINE omp_lock_t *CriticalLock() {return &criticalLock; }
  // init
  INLINE void InitTeamDescr();

 private:
  omptarget_nvptx_TaskDescr levelZeroTaskDescr ; // icv for team master initial thread
  omptarget_nvptx_WorkDescr workDescrForActiveParallel; // one, ONLY for the active par
  omp_lock_t criticalLock;
}; 


////////////////////////////////////////////////////////////////////////////////
// global data tables
////////////////////////////////////////////////////////////////////////////////

// aee: support only one kernel at at time on one device
extern __device__ omptarget_nvptx_TeamDescr omptarget_nvptx_teamContexts[MAX_NUM_TEAMS];

// aee: support only one kernel at at time on one device
extern __device__ omptarget_nvptx_ThreadPrivateContext omptarget_nvptx_threadPrivateContext;

extern __device__ omptarget_nvptx_GlobalICV omptarget_nvptx_globalICV;

////////////////////////////////////////////////////////////////////////////////
// get private data structures
////////////////////////////////////////////////////////////////////////////////

INLINE omptarget_nvptx_TeamDescr & getMyTeamDescriptor();
INLINE omptarget_nvptx_WorkDescr & getMyWorkDescriptor();
INLINE omptarget_nvptx_TaskDescr * getMyTopTaskDescriptor();
INLINE omptarget_nvptx_TaskDescr * getMyTopTaskDescriptor(int globalThreadId);


////////////////////////////////////////////////////////////////////////////////
// inlined implementation
////////////////////////////////////////////////////////////////////////////////

#include "supporti.h"
#include "omptarget-nvptxi.h"
#include "counter_groupi.h"

#endif

//===------------ libcall.cu - NVPTX OpenMP user calls ----------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP runtime functions that can be
// invoked by the user in an OpenMP region
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"
NOINLINE void PrintTaskDescr(omptarget_nvptx_TaskDescr *taskDescr, char *title, int level);

EXTERN double omp_get_wtick(void)
{
  double rc = omptarget_nvptx_globalICV.gpuCycleTime;
  PRINT(LD_IO, "call omp_get_wtick() returns %g\n", rc);
  return rc;
}

EXTERN double omp_get_wtime(void)
{
  double rc = omptarget_nvptx_globalICV.gpuCycleTime * clock64();
  PRINT(LD_IO, "call omp_get_wtime() returns %g\n", rc);
  return rc;
}

EXTERN void omp_set_num_threads(int num)
{
  PRINT(LD_IO, "call omp_set_num_threads(num %d)\n", num);
  if (num <= 0) {
     WARNING0(LW_INPUT, "expected positive num; ignore\n");
  } else {
    omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
    currTaskDescr->NThreads() = num;
  }
}

EXTERN int omp_get_num_threads(void)
{
  int gtid = GetGlobalThreadId();
  int rc = GetNumberOfOmpThreads(gtid);
  PRINT(LD_IO, "call omp_get_num_threads() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_max_threads(void)
{
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
  int rc = 1; // default is 1 thread avail
  if (! currTaskDescr->InParallelRegion()) {
    // not currently in a parallel region... all are available
    rc = GetNumberOfProcsInTeam();
    ASSERT0(LT_FUSSY, rc >= 0, "bad number of threads");
  }
  PRINT(LD_IO, "call omp_get_num_threads() return %\n", rc);
  return rc;
}

EXTERN int omp_get_thread_limit(void)
{
  // per contention group.. meaning threads in current team
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
  int rc = currTaskDescr->ThreadLimit();
  PRINT(LD_IO, "call omp_get_thread_limit() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_thread_num()
{ 
  int gtid = GetGlobalThreadId();
  int rc = GetOmpThreadId(gtid);
  PRINT(LD_IO, "call omp_get_thread_num() returns %d\n", rc);
  return rc;
}

EXTERN int omp_get_num_procs(void) 
{
  int rc = GetNumberOfThreadsInBlock();
  PRINT(LD_IO, "call omp_get_num_procs() returns %d\n", rc);
  return rc;
}

EXTERN int omp_in_parallel(void)
{
  int rc = 0;
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
  if (currTaskDescr->InParallelRegion()) {
    rc = 1;
  } 
  PRINT(LD_IO, "call omp_in_parallel() returns %d\n", rc);
  return rc;
}

EXTERN int omp_in_final(void)
{
  // treat all tasks as final... Specs may expect runtime to keep
  // track more precisely if a task was actively set by users... This
  // is not explicitely specified; will treat as if runtime can
  // actively decide to put a non-final task into a final one.
  int rc = 1;
  PRINT(LD_IO, "call omp_in_final() returns %d\n", rc);
  return rc;
}


EXTERN void omp_set_dynamic(int flag)
{
  PRINT(LD_IO, "call omp_set_dynamic(%d)\n", flag);

  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
  if (flag) {
    currTaskDescr->SetDynamic();
  } else {
    currTaskDescr->ClearDynamic();
  }
}

EXTERN int  omp_get_dynamic(void)
{
  int rc = 0;
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
  if (currTaskDescr->IsDynamic()) {
    rc = 1;
  }
  PRINT(LD_IO, "call omp_get_dynamic() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_nested(int flag)
{
  PRINT(LD_IO, "call omp_set_nested(%d) is ignored (no nested support)\n", flag);
}

EXTERN int omp_get_nested(void)
{
  int  rc = 0;
  PRINT(LD_IO, "call omp_get_nested() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_max_active_levels(int level)
{
  PRINT(LD_IO, "call omp_set_max_active_levels(%d) is ignored (no nested support)\n", level);
}

EXTERN int omp_get_max_active_levels(void)
{
  int  rc = 1;
  PRINT(LD_IO, "call omp_get_nested() returns %d\n", rc);
  return rc;
}

EXTERN int omp_get_level(void)
{
  int level = 0;
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
  ASSERT0(LT_FUSSY, currTaskDescr, "do not expect fct to be called in a non-active thread");
  do {
    if (currTaskDescr->IsParallelConstruct()) {
      level++;
    }
    currTaskDescr = currTaskDescr->GetPrevTaskDescr();
  } while (currTaskDescr);
  PRINT(LD_IO, "call omp_get_level() returns %d\n", level);
  return level;
}

EXTERN int omp_get_active_level(void)
{
  int level = 0; // no active level parallelism
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
  ASSERT0(LT_FUSSY, currTaskDescr, "do not expect fct to be called in a non-active thread");
  do {
    if (currTaskDescr->ThreadsInTeam() > 1) {
      // has a parallel with more than one thread in team
      level = 1;
      break;
    }
    currTaskDescr = currTaskDescr->GetPrevTaskDescr();
  } while (currTaskDescr);
  PRINT(LD_IO, "call omp_get_active_level() returns %d\n", level)
  return level;
}

EXTERN int omp_get_ancestor_thread_num(int level)
{
  int rc = 0;  // default at level 0
  if (level>=0) {
    int totLevel = omp_get_level();
    if (level<=totLevel) {
      omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
      int steps = totLevel - level;
      PRINT(LD_IO, "backtrack %d steps\n", steps);
      ASSERT0(LT_FUSSY, currTaskDescr, "do not expect fct to be called in a non-active thread");
      do {
        if (DON(LD_IOD)) PrintTaskDescr(currTaskDescr, (char *)"ancestor", steps);
        if (currTaskDescr->IsParallelConstruct()) {
	        // found the level
	        if (! steps) {
	          rc = currTaskDescr->ThreadId();
	          break;
	        }
          steps--;
	      }
	      currTaskDescr = currTaskDescr->GetPrevTaskDescr();
      } while (currTaskDescr);
      ASSERT0(LT_FUSSY, ! steps, "expected to find all steps");
    }
  }
  PRINT(LD_IO, "call omp_get_ancestor_thread_num(level %d) returns %d\n", level, rc)
  return rc;
}


EXTERN int omp_get_team_size(int level)
{
  int rc = 1;  // default at level 0
  if (level>=0) {
    int totLevel = omp_get_level();
    if (level<=totLevel) {
      omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
      int steps = totLevel - level;
      ASSERT0(LT_FUSSY, currTaskDescr, "do not expect fct to be called in a non-active thread");
      do {
        if (currTaskDescr->IsParallelConstruct()) {
	        if (! steps) {
	          // found the level
	          rc = currTaskDescr->ThreadsInTeam();
	          break;
	        }
          steps--;
	      }
	      currTaskDescr = currTaskDescr->GetPrevTaskDescr();
      } while (currTaskDescr);
      ASSERT0(LT_FUSSY, ! steps, "expected to find all steps");
    }
  }
  PRINT(LD_IO, "call omp_get_team_size(level %d) returns %d\n", level, rc)
  return rc;
}


EXTERN void omp_get_schedule(omp_sched_t * kind, int * modifier)
{
  omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
  *kind = currTaskDescr->GetRuntimeSched();
  *modifier = currTaskDescr->RuntimeChunkSize();
  PRINT(LD_IO, "call omp_get_schedule returns sched %d and modif %d\n",
    (int) *kind, *modifier);
}

EXTERN void omp_set_schedule(omp_sched_t kind, int modifier)
{
  PRINT(LD_IO, "call omp_set_schedule(sched %d, modif %d)\n",
    (int) kind, modifier);
  if (kind>=omp_sched_static && kind<omp_sched_auto) {
    omptarget_nvptx_TaskDescr *currTaskDescr = getMyTopTaskDescriptor();
    currTaskDescr->SetRuntimeSched(kind);
    currTaskDescr->RuntimeChunkSize() = modifier;
    PRINT(LD_IOD, "omp_set_schedule did set sched %d & modif %d\n",
      (int) currTaskDescr->GetRuntimeSched(), currTaskDescr->RuntimeChunkSize());

  }
}

EXTERN omp_proc_bind_t omp_get_proc_bind(void)
{
  PRINT0(LD_IO, "call omp_get_proc_bin() is true, regardless on state\n");
  return omp_proc_bind_true;
}

EXTERN int  omp_get_cancellation(void)
{
  int rc = omptarget_nvptx_globalICV.cancelPolicy;
  PRINT(LD_IO, "call omp_get_cancellation() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_default_device(int deviceId)
 {
  PRINT0(LD_IO, "call omp_get_default_device() is undef on device\n");
}

EXTERN int  omp_get_default_device(void)
{
  PRINT0(LD_IO, "call omp_get_default_device() is undef on device, returns 0\n");
  return 0;
}

EXTERN int  omp_get_num_devices(void)
{
  PRINT0(LD_IO, "call omp_get_num_devices() is undef on device, returns 0\n");
  return 0;
}

EXTERN int  omp_get_num_teams(void)
{
  int rc = GetNumberOfOmpTeams();
  PRINT(LD_IO, "call omp_get_num_teams() returns %d\n", rc);
  return rc;
}

EXTERN int omp_get_team_num()
{
  int rc = GetOmpTeamId();
  PRINT(LD_IO, "call omp_get_team_num() returns %d\n", rc);  
  return rc;
}

EXTERN int omp_is_initial_device(void)
{
  PRINT0(LD_IO, "call omp_is_initial_device() returns 0\n");
  return 0; // 0 by def on device
}

////////////////////////////////////////////////////////////////////////////////
// locks
////////////////////////////////////////////////////////////////////////////////

#define __OMP_SPIN 1000
#define UNSET 0
#define SET 1

EXTERN void omp_init_lock(omp_lock_t * lock)
{
  *lock = UNSET;
  PRINT0(LD_IO, "call omp_init_lock()\n");
}

EXTERN void omp_destroy_lock(omp_lock_t * lock)
{
  PRINT0(LD_IO, "call omp_destroy_lock()\n");
}

EXTERN void omp_set_lock(omp_lock_t * lock)
{
  // int atomicCAS(int* address, int compare, int val);
   // (old == compare ? val : old)
   int compare = UNSET;
   int val = SET;

   // TODO: not sure spinning is a good idea here..
   while (atomicCAS(lock, compare, val) != UNSET) {
   
     clock_t start = clock();
     clock_t now;
     for (;;)
     {
       now = clock();
       clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
       if (cycles >= __OMP_SPIN*blockIdx.x) {
         break;
       }
     }
   } // wait for 0 to be the read value
    
  PRINT0(LD_IO, "call omp_set_lock()\n");
}

EXTERN void omp_unset_lock(omp_lock_t * lock)
{
  int compare = SET;
  int val = UNSET;
  int old = atomicCAS (lock, compare, val);

  PRINT0(LD_IO, "call omp_unset_lock()\n");
}

EXTERN int omp_test_lock(omp_lock_t * lock)
{
  // int atomicCAS(int* address, int compare, int val);
  // (old == compare ? val : old)
  int compare = UNSET;
  int val = SET;
  
  // TODO: should check for the lock to be SET?
  int ret = atomicCAS(lock, compare, val);
  
  PRINT(LD_IO, "call omp_test_lock() return %d\n", ret);
  
  return ret;
}

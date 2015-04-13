//===------------ option.h - NVPTX OpenMP GPU options ------------ CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// GPU default options
//
//===----------------------------------------------------------------------===//
#ifndef _OPTION_H_
#define _OPTION_H_ 

////////////////////////////////////////////////////////////////////////////////
// Kernel options
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// following two defs must match absolute limit hardwired in the host RTL
#define TEAMS_ABSOLUTE_LIMIT 512 /* omptx limit (must match teamsAbsoluteLimit) */
#define THREAD_ABSOLUTE_LIMIT 1024 /* omptx limit (must match threadAbsoluteLimit) */

// max number of blocks depend on the kernel we are executing - pick default here
#define MAX_NUM_TEAMS TEAMS_ABSOLUTE_LIMIT
#define WARPSIZE 32
#define MAX_NUM_WARPS MAX_NUM_TEAMS * THREAD_ABSOLUTE_LIMIT
#define MAX_NUM_THREADS MAX_NUM_WARPS * WARPSIZE

#ifdef OMPTHREAD_IS_WARP
  // assume here one OpenMP thread per CUDA warp
  #define MAX_NUM_OMP_THREADS MAX_NUM_WARPS
#else
  // assume here one OpenMP thread per CUDA thread
  #define MAX_NUM_OMP_THREADS MAX_NUM_THREADS
#endif

////////////////////////////////////////////////////////////////////////////////
// algo options
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// data options
////////////////////////////////////////////////////////////////////////////////


// decide if counters are 32 or 64 bit
#define Counter unsigned long long

// aee: KMP defines kmp_int to be 32 or 64 bits depending on the target. 
// think we don't need it here (meaning we can be always 64 bit compatible)
/*
#ifdef KMP_I8
  typedef kmp_int64		kmp_int;
#else
  typedef kmp_int32		kmp_int;
#endif
*/

////////////////////////////////////////////////////////////////////////////////
// misc options (by def everythig here is device)
////////////////////////////////////////////////////////////////////////////////

#define EXTERN extern "C" __device__
#define INLINE __inline__ __device__
#define NOINLINE __noinline__ __device__
#ifndef TRUE
  #define TRUE 1
#endif
#ifndef FALSE
  #define FALSE 0
#endif

#endif

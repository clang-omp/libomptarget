//===------------ sync.h - NVPTX OpenMP synchronizations --------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Include all synchronization.
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

////////////////////////////////////////////////////////////////////////////////
// KMP Ordered calls
////////////////////////////////////////////////////////////////////////////////


EXTERN void __kmpc_ordered (kmp_Indent * loc, int32_t gtid)
{
  PRINT0(LD_IO, "call kmpc_ordered\n");
}


EXTERN void __kmpc_end_ordered (kmp_Indent * loc, int32_t gtid)
{
  PRINT0(LD_IO, "call kmpc_end_ordered\n");
}



////////////////////////////////////////////////////////////////////////////////
// KMP Barriers
////////////////////////////////////////////////////////////////////////////////

// a team is a block: we can use CUDA native synchronization mechanism
// FIXME: what if not all threads (warps) participate to the barrier?
// We may need to implement it differently

EXTERN int32_t __kmpc_cancel_barrier (kmp_Indent* loc_ref, int32_t gtid)
{
  PRINT0(LD_IO, "call kmpc_cancel_barrier\n");
  __syncthreads();	
  PRINT0(LD_SYNC, "completed kmpc_cancel_barrier\n");
  return 0;
}

// aee this one shoud be discontinued
EXTERN void __kmpc_barrier (kmp_Indent* loc_ref, int32_t gtid)
{
  PRINT0(LD_IO, "call kmpc_barrier\n");
  __syncthreads();
  PRINT0(LD_SYNC, "completed kmpc_barrier\n");
}

////////////////////////////////////////////////////////////////////////////////
// KMP MASTER
////////////////////////////////////////////////////////////////////////////////

INLINE int32_t IsMaster()
{
  // only the team master updates the state
  int gtid = GetGlobalThreadId();
  int ompThreadId = GetOmpThreadId(gtid);
  return IsTeamMaster(ompThreadId);
}

EXTERN int32_t __kmpc_master(kmp_Indent *loc, int32_t global_tid)
{
  PRINT0(LD_IO, "call kmpc_master\n");
  return IsMaster();
}

EXTERN void  __kmpc_end_master(kmp_Indent *loc, int32_t global_tid)
{
  PRINT0(LD_IO, "call kmpc_end_master\n");
  ASSERT0(LT_FUSSY, IsMaster(), "expected only master here");
}

////////////////////////////////////////////////////////////////////////////////
// KMP SINGLE
////////////////////////////////////////////////////////////////////////////////

EXTERN int32_t __kmpc_single(kmp_Indent *loc, int32_t global_tid)
{
  PRINT0(LD_IO, "call kmpc_single\n");
  // decide to implement single with master; master get the single
  return IsMaster();
}

EXTERN void __kmpc_end_single(kmp_Indent *loc, int32_t global_tid)
{
  PRINT0(LD_IO, "call kmpc_end_single\n");
  // decide to implement single with master: master get the single
  ASSERT0(LT_FUSSY, IsMaster(), "expected only master here");
  // sync barrier is explicitely called... so that is not a problem
}

////////////////////////////////////////////////////////////////////////////////
// Flush
////////////////////////////////////////////////////////////////////////////////

EXTERN void __kmpc_flush(kmp_Indent *loc)
{
  PRINT0(LD_IO, "call kmpc_flush\n");
  // not aware of anything to do for flush
}

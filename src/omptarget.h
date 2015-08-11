//===-------- omptarget.h - Target independent OpenMP target RTL -- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to be used by Clang during the codegen of a
// target region.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#ifndef _OMPTARGET_H_
#define _OMPTARGET_H_

#define OFFLOAD_SUCCESS (0)
#define OFFLOAD_FAIL    (~0)

#define OFFLOAD_DEVICE_DEFAULT     -1
#define OFFLOAD_DEVICE_CONSTRUCTOR -2
#define OFFLOAD_DEVICE_DESTRUCTOR  -3

/// Data attributes for each data reference used in an OpenMP target region.
enum tgt_map_type{
  tgt_map_alloc     = 0x00,   // allocate memory in the device for this reference
  tgt_map_to        = 0x01,   // copy the data to the device but do not update the host memory
  tgt_map_from      = 0x02,   // copy the data to the host but do not update the device memory
  tgt_map_always    = 0x04,
  tgt_map_release   = 0x08,
  tgt_map_delete    = 0x18,
  tgt_map_pointer   = 0x20,
  tgt_map_extra     = 0x40
};

/// This struct is a record of an entry point or global. For a function
/// entry point the size is expected to be zero
struct __tgt_offload_entry{
  void      *addr;       // Pointer to the offload entry info (function or global)
  char      *name;       // Name of the function or global
  int64_t   size;        // Size of the entry info (0 if it a function)
};

/// This struct is a record of the device image information
struct __tgt_device_image{
  void   *ImageStart;       // Pointer to the target code start
  void   *ImageEnd;         // Pointer to the target code end
  __tgt_offload_entry  *EntriesBegin;   // Begin of the table with all the entries
  __tgt_offload_entry  *EntriesEnd;     // End of the table with all the entries (non inclusive)
};

/// This struct is a record of all the host code that may be offloaded to a target.
struct __tgt_bin_desc{
  int32_t              NumDevices;      // Number of devices supported
  __tgt_device_image   *DeviceImages;   // Arrays of device images (one per device)
  __tgt_offload_entry  *EntriesBegin;   // Begin of the table with all the entries
  __tgt_offload_entry  *EntriesEnd;     // End of the table with all the entries (non inclusive)
};

/// This struct contains the offload entries identified by the target runtime
struct __tgt_target_table{
  __tgt_offload_entry  *EntriesBegin;   // Begin of the table with all the entries
  __tgt_offload_entry  *EntriesEnd;     // End of the table with all the entries (non inclusive)
};

#ifdef __cplusplus
extern "C" {
#endif

void omp_set_default_device(int device_num);
int omp_get_default_device(void);
int omp_get_num_devices(void);
int omp_is_initial_device(void);

/// adds a target shared library to the target execution image
void __tgt_register_lib(__tgt_bin_desc *desc);

/// removes a target shared library to the target execution image
void __tgt_unregister_lib(__tgt_bin_desc *desc);

// creates host to the target data mapping, store it in the
// libtarget.so internal structure (an entry in a stack of data maps) and
// passes the data to the device;
int __tgt_target_data_begin(int32_t device_id, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types);
int __tgt_target_data_begin_nowait(int32_t device_id, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types,
  int32_t depNum, void * depList, int32_t noAliasDepNum, void * noAliasDepList);

// passes data from the target, release target memory and destroys the
// host-target mapping (top entry from the stack of data maps) created by
// the last __tgt_target_data_begin
int __tgt_target_data_end(int32_t device_id, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types);
int __tgt_target_data_end_nowait(int32_t device_id, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types,
  int32_t depNum, void * depList, int32_t noAliasDepNum, void * noAliasDepList);

/// passes data to/from the target
int __tgt_target_data_update(int32_t device_id, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types);
int __tgt_target_data_update_nowait(int32_t device_id, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types,
  int32_t depNum, void * depList, int32_t noAliasDepNum, void * noAliasDepList);

// performs the same actions as data_begin in case arg_num is non-zero
// and initiates run of offloaded region on target platform; if arg_num
// is non-zero after the region execution is done it also performs the
// same action as data_update and data_end aboveThe following types are
// used; this function return 0 if it was able to transfer the execution
// to a target and an int different from zero otherwise
int __tgt_target(int32_t device_id, void *host_ptr, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types);
int __tgt_target_nowait(int32_t device_id, void *host_ptr, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types,
  int32_t depNum, void * depList, int32_t noAliasDepNum, void * noAliasDepList);

int __tgt_target_teams(int32_t device_id, void *host_ptr, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types,
  int32_t num_teams, int32_t thread_limit);
int __tgt_target_teams_nowait(int32_t device_id, void *host_ptr, int32_t arg_num,
  void** args_base, void **args, int64_t *arg_sizes, int32_t *arg_types,
  int32_t num_teams, int32_t thread_limit,
  int32_t depNum, void * depList, int32_t noAliasDepNum, void * noAliasDepList);

#ifdef __cplusplus
}
#endif

#ifdef OMPTARGET_DEBUG
# define DEBUGP(prefix, ...) { fprintf(stderr, "%s --> ", prefix); \
                             fprintf(stderr, __VA_ARGS__); }
#else
# define DEBUGP(prefix, ...) {}
#endif

#ifdef __cplusplus
  #define EXTERN extern "C"
#else
  #define EXTERN extern
#endif

#endif // _OMPTARGET_H_

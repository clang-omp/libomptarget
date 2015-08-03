//===----RTLs/cuda/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// RTL for CUDA machine
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gelf.h>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>

#include "omptarget.h"

#ifndef TARGET_NAME
#define TARGET_NAME Generic-64bit
#endif

#define GETNAME2(name) #name
#define GETNAME(name) GETNAME2(name)
#define DP(...) DEBUGP("Target " GETNAME(TARGET_NAME) " RTL",__VA_ARGS__)

// Utility for retrieving and printing CUDA error string
#ifdef CUDA_ERROR_REPORT
#define CUDA_ERR_STRING(err)               \
    do {                                   \
        const char* errStr;                \
        cuGetErrorString (err, &errStr);   \
        DP("CUDA error is: %s\n", errStr); \
    } while (0)
#else
#define CUDA_ERR_STRING(err) {}
#endif

// NVPTX image start encodes a struct that also includes the host entries begin
// and end pointers. The host entries are used by the runtime to accelerate
// the retrieval of the target entry pointers
struct __tgt_nvptx_device_image_start{
  void                  *RealStart;        // Pointer to actual NVPTX elf image
  char                  *TgtName;          // Name of the target of the image
  __tgt_offload_entry   *HostStart;        // Pointer to the host entries start
  __tgt_offload_entry   *HostEnd;          // Pointer to the host entries end
};

/// Account the memory allocated per device
struct AllocMemEntryTy{
  int64_t TotalSize;
  std::vector<std::pair<void*,int64_t> > Ptrs;

  AllocMemEntryTy() : TotalSize(0) {}
};

/// Keep entries table per device
struct FuncOrGblEntryTy{
  __tgt_target_table Table;
  std::vector<__tgt_offload_entry> Entries;
};

/// Use a single entity to encode a kernel and a set of flags
struct KernelTy{
  CUfunction Func;
  int SimdInfo;

  // keep track of cuda pointer to write to it when thread_limit value
  // changes (check against last value written to ThreadLimit
  CUdeviceptr ThreadLimitPtr;
  int ThreadLimit;

  KernelTy(CUfunction _Func, int _SimdInfo, CUdeviceptr _ThreadLimitPtr)
    : Func(_Func), SimdInfo(_SimdInfo), ThreadLimitPtr(_ThreadLimitPtr) {
    ThreadLimit = 0; //default (0) signals that it was not initialized
  };
};

/// List that contains all the kernels.
/// FIXME: we may need this to be per device and per library.
std::list<KernelTy> KernelsList;

/// Class containing all the device information
class RTLDeviceInfoTy{
  std::vector<FuncOrGblEntryTy> FuncGblEntries;

public:
  int NumberOfDevices;
  std::vector<CUmodule> Modules;
  std::vector<CUcontext> Contexts;
  std::vector<int> ThreadsPerBlock;
  std::vector<int> BlocksPerGrid;

  // Record entry point associated with device
  void addOffloadEntry(int32_t device_id, __tgt_offload_entry entry ){
    assert( device_id < FuncGblEntries.size() && "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id];

    E.Entries.push_back(entry);
  }

  // Return true if the entry is associated with device
  bool findOffloadEntry(int32_t device_id, void *addr){
    assert( device_id < FuncGblEntries.size() && "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id];

    for(unsigned i=0; i<E.Entries.size(); ++i){
      if(E.Entries[i].addr == addr)
        return true;
    }

    return false;
  }

  // Return the pointer to the target entries table
  __tgt_target_table *getOffloadEntriesTable(int32_t device_id){
    assert( device_id < FuncGblEntries.size() && "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id];

    int32_t size = E.Entries.size();

    // Table is empty
    if(!size)
      return 0;

    __tgt_offload_entry *begin = &E.Entries[0];
    __tgt_offload_entry *end = &E.Entries[size-1];

    // Update table info according to the entries and return the pointer
    E.Table.EntriesBegin = begin;
    E.Table.EntriesEnd = ++end;

    return &E.Table;
  }

  // Clear entries table for a device
  void clearOffloadEntriesTable(int32_t device_id){
    assert( device_id < FuncGblEntries.size() && "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id];
    E.Entries.clear();
    E.Table.EntriesBegin = E.Table.EntriesEnd = 0;
  }

  RTLDeviceInfoTy(){
    DP ("Start initializing CUDA\n");

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS)
    {
      DP ("Error when initializing CUDA\n");
      CUDA_ERR_STRING(err);
      return;
    }

    NumberOfDevices = 0;

    err = cuDeviceGetCount(&NumberOfDevices);
    if (err != CUDA_SUCCESS)
    {
      DP ("Error when getting CUDA device count\n");
      CUDA_ERR_STRING(err);
      return;    
    }    

    if (NumberOfDevices == 0){
      DP("There are no devices supporting CUDA.\n");
      return;
    }

    FuncGblEntries.resize(NumberOfDevices);
    Contexts.resize(NumberOfDevices);
    ThreadsPerBlock.resize(NumberOfDevices);
    BlocksPerGrid.resize(NumberOfDevices);
  }

  ~RTLDeviceInfoTy(){
    // Close modules
    for(int32_t i=0; i<Modules.size(); ++i )
      if(Modules[i])
      {
        CUresult err = cuModuleUnload(Modules[i]);
	if (err != CUDA_SUCCESS)
	{
	  DP ("Error when unloading CUDA module\n");
	  CUDA_ERR_STRING(err);
	}
      }

    // Destroy contexts
    for(int32_t i=0; i<Contexts.size(); ++i )
      if(Contexts[i])
      {
        CUresult err = cuCtxDestroy(Contexts[i]);
	if (err != CUDA_SUCCESS)
	{
	  DP ("Error when destroying CUDA context\n");
	  CUDA_ERR_STRING(err);
	}
      }
  }
};

static RTLDeviceInfoTy DeviceInfo;

#ifdef __cplusplus
extern "C" {
#endif

int __tgt_rtl_device_type(int32_t device_id){

  if( device_id < DeviceInfo.NumberOfDevices)
    return 190; // EM_CUDA

  return 0;
}

int __tgt_rtl_number_of_devices(){
  return DeviceInfo.NumberOfDevices;
}

int32_t __tgt_rtl_init_device(int32_t device_id){

  CUdevice cuDevice;
  DP ("Getting device %d\n", device_id);
  CUresult err = cuDeviceGet(&cuDevice, device_id);
  if (err != CUDA_SUCCESS)
  {
    DP ("Error when getting CUDA device with id = %d\n", device_id);
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  // Create the context and save it to use whenever this device is selected
  CUcontext cuContext;
  err = cuCtxCreate(&DeviceInfo.Contexts[device_id], CU_CTX_SCHED_BLOCKING_SYNC, cuDevice);
  if (err != CUDA_SUCCESS)
  {
    DP ("Error when creating a CUDA context\n");
    CUDA_ERR_STRING(err);
    return OFFLOAD_FAIL;
  }

  // scan properties to determine number of threads/blocks per block/grid
  struct cudaDeviceProp Properties;
  cudaError_t error = cudaGetDeviceProperties(&Properties, device_id);
  if (error != cudaSuccess) {
    DP("Error when getting device Properties, use default\n");
    DeviceInfo.BlocksPerGrid[device_id] = 32;
    DeviceInfo.ThreadsPerBlock[device_id] = 512;
  } else {
    DeviceInfo.BlocksPerGrid[device_id] = Properties.multiProcessorCount;
    // exploit threads only along x axis
    DeviceInfo.ThreadsPerBlock[device_id] = Properties.maxThreadsDim[0];
    if (Properties.maxThreadsDim[0] < Properties.maxThreadsPerBlock) {
      DP("use up to %d threads, fewer than max per blocks along xyz %d\n",
        Properties.maxThreadsDim[0], Properties.maxThreadsPerBlock);
    }
  }
  DP("Default number of blocks %d & threads %d\n", 
    DeviceInfo.BlocksPerGrid[device_id], DeviceInfo.ThreadsPerBlock[device_id]);

  // done
  return OFFLOAD_SUCCESS;
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id, __tgt_device_image *image){

  //Set the context we are using
  CUresult err = cuCtxSetCurrent (DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS)
  {
    DP ("Error when setting a CUDA context for device %d\n", device_id);
    CUDA_ERR_STRING(err);
    return NULL;
  }

  //Clear the offload table as we are going to create a new one
  DeviceInfo.clearOffloadEntriesTable(device_id);

  // Create the module and extract the function pointers

  CUmodule cumod;
  DP("load data from image %llx\n", (unsigned long long) image->ImageStart);
  err = cuModuleLoadDataEx (&cumod, image->ImageStart, 0, NULL, NULL);
  if (err != CUDA_SUCCESS)
  {
    DP ("Error when loading CUDA module\n");
    CUDA_ERR_STRING (err);
    return NULL;
  }

  DP ("CUDA module successfully loaded!\n");
  DeviceInfo.Modules.push_back(cumod);

  // Here, we take advantage of the data that is appended after img_end to get
  // the symbols' name we need to load. This data consist of the host entries
  // begin and end as well as the target name (see the offloading linker script
  // creation in clang compiler).
  // Find the symbols in the module by name. The name can be obtain by
  // concatenating the host entry name with the target name

  __tgt_offload_entry *HostBegin = image->EntriesBegin;
  __tgt_offload_entry *HostEnd   = image->EntriesEnd;

  for( __tgt_offload_entry *e = HostBegin; e != HostEnd; ++e) {

    if( !e->addr ){
      // FIXME: Probably we should fail when something like this happen, the
      // host should have always something in the address to uniquely identify
      // the target region.
      DP("Analyzing host entry '<null>' (size = %lld)...\n",
         (unsigned long long)e->size);

      __tgt_offload_entry entry = *e;
      DeviceInfo.addOffloadEntry(device_id, entry);
      continue;
    }

    if( e->size ){

      __tgt_offload_entry entry = *e;

      CUdeviceptr cuptr;
      size_t cusize;
      err = cuModuleGetGlobal(&cuptr,&cusize,cumod,e->name);

      if (err != CUDA_SUCCESS){
        DP("loading global '%s' (Failed)\n",e->name);
        CUDA_ERR_STRING (err);
        return NULL;
      }

      if ((int32_t)cusize != e->size){
        DP("loading global '%s' - size mismatch (%lld != %lld)\n",e->name,
            (unsigned long long)cusize,
            (unsigned long long)e->size);
        CUDA_ERR_STRING (err);
        return NULL;
      }

      DP("Entry point %ld maps to global %s (%016lx)\n",e-HostBegin,e->name,(long)cuptr);
      entry.addr = (void*)cuptr;

      DeviceInfo.addOffloadEntry(device_id, entry);

      continue;
    }

    CUfunction fun;
    err = cuModuleGetFunction (&fun, cumod, e->name);

    if (err != CUDA_SUCCESS){
      DP("loading '%s' (Failed)\n",e->name);
      CUDA_ERR_STRING (err);
      return NULL;
    }

    DP("Entry point %ld maps to %s (%016lx)\n",e-HostBegin,e->name,(Elf64_Addr)fun);

    // default value
    int8_t SimdInfoVal = 1;

    // obtain and save simd_info value for target region
    const char suffix[] = "_simd_info";
    char * SimdInfoName = (char *) malloc((strlen(e->name)+strlen(suffix))*
        sizeof(char));
    sprintf(SimdInfoName, "%s%s", e->name, suffix);

    CUdeviceptr SimdInfoPtr;
    size_t cusize;
    err = cuModuleGetGlobal(&SimdInfoPtr,&cusize,cumod,SimdInfoName);
    if (err == CUDA_SUCCESS) {
      if ((int32_t)cusize != sizeof(int8_t)){
        DP("loading global simd_info '%s' - size mismatch (%lld != %lld)\n", SimdInfoName, (unsigned long long)cusize,(unsigned long long)sizeof(int8_t));
        CUDA_ERR_STRING (err);
        return NULL;
      }

      err = cuMemcpyDtoH(&SimdInfoVal,(CUdeviceptr)SimdInfoPtr,cusize);
      if (err != CUDA_SUCCESS)
      {
        DP("Error when copying data from device to host. Pointers: "
            "host = 0x%016lx, device = 0x%016lx, size = %lld\n",(Elf64_Addr)&SimdInfoVal, (Elf64_Addr)SimdInfoPtr,(unsigned long long)cusize);
        CUDA_ERR_STRING (err);
        return NULL;
      }
      if (SimdInfoVal < 1) {
        DP("Error wrong simd_info value specified in cubin file: %d\n", SimdInfoVal);
        return NULL;
      }
    }

    // obtain cuda pointer to global tracking thread limit
    const char SuffixTL[] = "_thread_limit";
    char * ThreadLimitName = (char *) malloc((strlen(e->name)+strlen(SuffixTL))*
        sizeof(char));
    sprintf(ThreadLimitName, "%s%s", e->name, SuffixTL);

    CUdeviceptr ThreadLimitPtr;
    err = cuModuleGetGlobal(&ThreadLimitPtr,&cusize,cumod,ThreadLimitName);
    if (err != CUDA_SUCCESS) {
      DP("retrieving pointer for %s global\n", ThreadLimitName);
      CUDA_ERR_STRING (err);
      return NULL;
    }
    if ((int32_t)cusize != sizeof(int32_t)) {
      DP("loading global thread_limit '%s' - size mismatch (%lld != %lld)\n", ThreadLimitName, (unsigned long long)cusize,(unsigned long long)sizeof(int32_t));
      CUDA_ERR_STRING (err);
      return NULL;
    }

    // encode function and kernel
    KernelsList.push_back(KernelTy(fun, SimdInfoVal, ThreadLimitPtr));

    __tgt_offload_entry entry = *e;
    entry.addr = (void*)&KernelsList.back();
    DeviceInfo.addOffloadEntry(device_id, entry);
  }

  return DeviceInfo.getOffloadEntriesTable(device_id);
}

void *__tgt_rtl_data_alloc(int32_t device_id, int64_t size){

  //Set the context we are using  
  CUresult err = cuCtxSetCurrent (DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS)
  {
    DP("Error while trying to set CUDA current context\n");
    CUDA_ERR_STRING (err);
    return NULL;
  }

  CUdeviceptr ptr;
  err = cuMemAlloc(&ptr, size);
  if (err != CUDA_SUCCESS)
  {
    DP("Error while trying to allocate %d\n", err);
    CUDA_ERR_STRING (err);
    return NULL;
  }

  void *vptr = (void*) ptr;
  return vptr;
}

int32_t __tgt_rtl_data_submit(int32_t device_id, void *tgt_ptr, void *hst_ptr, int64_t size){
  //Set the context we are using
  CUresult err = cuCtxSetCurrent (DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS)
  {
    DP("Error when setting CUDA context\n");
    CUDA_ERR_STRING (err);
    return OFFLOAD_FAIL;
  }

  err = cuMemcpyHtoD((CUdeviceptr)tgt_ptr, hst_ptr, size);
  if (err != CUDA_SUCCESS)
  {
    DP("Error when copying data from host to device. Pointers: "
      "host = 0x%016lx, device = 0x%016lx, size = %lld\n",
       (Elf64_Addr)hst_ptr, (Elf64_Addr)tgt_ptr, (unsigned long long)size);
    CUDA_ERR_STRING (err);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_retrieve(int32_t device_id, void *hst_ptr, void *tgt_ptr, int64_t size){
  //Set the context we are using
  CUresult err = cuCtxSetCurrent (DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS)
  {
    DP("Error when setting CUDA context\n");
    CUDA_ERR_STRING (err);
    return OFFLOAD_FAIL;
  }

  err = cuMemcpyDtoH(hst_ptr,(CUdeviceptr)tgt_ptr,size);
  if (err != CUDA_SUCCESS)
  {
    DP("Error when copying data from device to host. Pointers: "
      "host = 0x%016lx, device = 0x%016lx, size = %lld\n",
        (Elf64_Addr)hst_ptr, (Elf64_Addr)tgt_ptr, (unsigned long long)size);
    CUDA_ERR_STRING (err);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_delete(int32_t device_id, void* tgt_ptr){
  //Set the context we are using
  CUresult err = cuCtxSetCurrent (DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS)
  {
    DP("Error when setting CUDA context\n");
    CUDA_ERR_STRING (err);
    return OFFLOAD_FAIL;
  }

  err = cuMemFree((CUdeviceptr)tgt_ptr);
  if (err != CUDA_SUCCESS)
  {
    DP("Error when freeing CUDA memory\n");
    CUDA_ERR_STRING (err);
    return OFFLOAD_FAIL;
  }
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_team_region(int32_t device_id, 
  void *tgt_entry_ptr, void **tgt_args, int32_t arg_num,
  int32_t team_num, int32_t thread_limit)
{
  //Set the context we are using
  CUresult err = cuCtxSetCurrent (DeviceInfo.Contexts[device_id]);
  if (err != CUDA_SUCCESS)
  {
    DP("Error when setting CUDA context\n");
    CUDA_ERR_STRING (err);
    return OFFLOAD_FAIL;
  }

  // All args are references
  std::vector<void*> args(arg_num);

  for(int32_t i=0; i<arg_num; ++i)
    args[i] = &tgt_args[i];

  KernelTy *KernelInfo = (KernelTy*)tgt_entry_ptr;

  int cudaThreadsPerBlock = (thread_limit<=0 ||
    thread_limit*KernelInfo->SimdInfo > DeviceInfo.ThreadsPerBlock[device_id])?
        DeviceInfo.ThreadsPerBlock[device_id] :
        thread_limit*KernelInfo->SimdInfo;

  // update thread limit content in gpu memory if un-initialized or changed
  if (KernelInfo->ThreadLimit == 0 || KernelInfo->ThreadLimit != thread_limit) {
    // always capped by maximum number of threads in a block: even if 1 OMP thread
    // is 1 independent CUDA thread, we may have up to max block size OMP threads
    // if the user request thread_limit(tl) with tl > max block size, we
    // only start max block size CUDA threads
    if (thread_limit > DeviceInfo.ThreadsPerBlock[device_id])
    thread_limit = DeviceInfo.ThreadsPerBlock[device_id];

    KernelInfo->ThreadLimit = thread_limit;
    err = cuMemcpyHtoD(KernelInfo->ThreadLimitPtr,&thread_limit,sizeof(int32_t));

    if (err != CUDA_SUCCESS) {
      DP("Error when setting thread limit global\n");
      return OFFLOAD_FAIL;
    }
  }

  int blocksPerGrid = team_num>0 ? team_num : 
    DeviceInfo.BlocksPerGrid[device_id];
  int nshared = 0;

  // Run on the device
  DP("launch kernel with %d blocks and %d threads\n", blocksPerGrid, cudaThreadsPerBlock);

  err = cuLaunchKernel(KernelInfo->Func,
    blocksPerGrid, 1, 1, cudaThreadsPerBlock, 1, 1, nshared, 0, &args[0], 0);
  if( err != CUDA_SUCCESS )
  {
    DP("Device kernel launching failed!\n");
    CUDA_ERR_STRING (err);
    assert(err == CUDA_SUCCESS && "Unable to launch target execution!" );
    return OFFLOAD_FAIL;
  }

  DP("Execution of entry point at %016lx successful!\n",
    (Elf64_Addr)tgt_entry_ptr);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr, 
  void **tgt_args, int32_t arg_num)
{
  // use one team and one thread
  // fix thread num
  int32_t team_num = 1;
  int32_t thread_limit = 0; // use default
  return __tgt_rtl_run_target_team_region(device_id, 
    tgt_entry_ptr, tgt_args, arg_num, team_num, thread_limit);
}


#ifdef __cplusplus
}
#endif

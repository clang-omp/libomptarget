# libomptarget - OpenMP offloading runtime libraries for Clang 

This is a prototype implementation of the OpenMP offloading library to be 
supported by Clang. The current implementation has been tested in Linux so far.

The current implementation of this library can be classified into three 
components: target agnostic offloading, target specific offloading plugins, and 
target specific runtime library.   

In order to build the libraries run:

```
make
```

or

```
mkdir build
cd build
cmake ..
make
```

Both build systems are prepared to detect the host machine automatically as well
as the target devices by looking for the correspondent toolkit (e.g. CUDA). If a 
supported toolkit is detect, the makefiles will create a library for it. 
However, it is possible that some systems require adjustments in the look-up 
paths.   

All the libraries will be created in ./lib folder. Typically, you will get:

libomptarget.so - The main and target agnostic library

libomptarget.rtl.[toolkit name].so - The target specific plugins. These plugins 
are loaded at runtime by libomptarget.so to interact with a given device.

libomptarget-[target name].[a,so] - The target specific runtime libraries. These 
libraries should be passed to the target linker as they implement the runtime 
calls produced by Clang during code generation. 

Note that the interface of all the libraries in this project  is likely to 
change in the future.

# Target agnostic offloading - libomptarget.so

This component contains the logic to launch the initialization of the devices 
supported by the current program, create device data environments and launch 
executions of kernels (OpenMP target regions). In order to deal with a specific 
device this component detects and loads the corresponding plugin. 

This component has been tested for:

  - powerpc64-ibm-linux-gnu 
  - powerpc64le-ibm-linux-gnu
  - x86_64-pc-linux-gnu
               
The code of this component is under ./src

# Target specific plugins - libomptarget.rtl.[toolkit name].so
 
These plugins are used by libomptarget.so to deal with a given target. They all 
use the same interface and implement basic functionality like device 
initialization, data movement to/from device and kernel launching.
 
The current implementation supports the following plugins:
  - generic 64-bit - this implementation is suitable for powerpc64, powerpc64le 
  and x86_64 target
  
  - cuda - plugin for Nvidia GPUs implemented on top of the CUDA device runtime 
  library
                          
The code for this component is under ./RTLs

# Target specific runtime libraries - libomptarget-[target name].[a,so]
                          
These libraries implement the OpenMP runtime calls used by a given device during 
execution.

The current implementation includes a library for:
  - nvptx: library written in CUDA for Nvidia GPUs. Tested with CUDA compilation 
  tools V7.0.27. The CUDA architecture can be set using cmake by setting 
  OMPTARGET\_NVPTX\_SM to a comma separated list of target architectures. For 
  example, to compile for sm\_30 and sm\_35 one can define 
  `-DOMPTARGET_NVPTX_SM=30,35` when calling cmake. If not using cmake the same 
  goal can be achieved by passing `OMPTARGET_NVPTX_SM=30,35` to make. In order 
  to use this library with Clang the user has to set LIBRARY_PATH to point to 
  ./lib so that Clang passes the right information to the target linker.
       
For powerpc64, powerpc64le and x86_64 devices, existing host runtime libraries 
(e.g. openmp.llvm.org) can be used for when these devices are used as OpenMP 
targets.
        
The code for this component is under ./DevRTLs

# TensorFlow on intel GPU
 
 - oneAPI software stack
![](https://spec.oneapi.io/level-zero/latest/_images/one_api_sw_stack.png)


**Comprehensive Tutorials**

[Source code](https://github.com/oneapi-src/oneAPI-samples/tree/master/Publications)

- [Data Parallel C++: Mastering DPC++ for Programming of Heterogeneous Systems using C++ and SYCL](https://www.apress.com/gp/book/9781484255735)
- [oneAPI GPU Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top.html)



## TensorFlow(Framework)

 - [Next PluggableDevice](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/common_runtime/next_pluggable_device)

RFCs

- [Modular TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md)
- [~~StreamExecutor C API~~](https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api.md)
- [Pluggable device for TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20200624-pluggable-device-for-tensorflow.md)
- [Kernel and Op Implementation and Registration API](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md)
- [Kernel Extension for Variable Operations API](https://github.com/tensorflow/community/blob/dd3c8761213043a543fc3665949ab901f86b26f9/rfcs/20210504-kernel-extension-variable-ops.md)
- [Modular TensorFlow Graph C API](https://github.com/tensorflow/community/blob/master/rfcs/20201027-modular-tensorflow-graph-c-api.md)



## SYCL(Programming model)

- [SYCL spec](https://www.khronos.org/registry/SYCL/)



### DPC++ Compiler

- [DPC++ Compiler Developer Guide and Reference](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top.html)
- [source code](https://github.com/intel/llvm/tree/sycl/sycl)
  - [DPC++ Compiler and Runtime architecture design](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/CompilerAndRuntimeDesign.md)
  - [Environment Variables](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md)
  - [~~Level Zero plugin~~](https://github.com/intel/llvm/tree/sycl/sycl/plugins/level_zero)
  - [Unified Runtime](https://github.com/intel/llvm/tree/sycl/sycl/plugins/unified_runtime)
  - [DPC++ extensions to SYCL specification](https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions)



## [Level Zero(Device runtime)](https://spec.oneapi.com/level-zero/latest/index.html)



### Driver

- [Intel Graphics Compute Runtime](https://github.com/intel/compute-runtime)
  - [debug environment variables](https://github.com/intel/compute-runtime/blob/master/shared/source/debug_settings/debug_variables_base.inl)
- [Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler)
  - [Execution Model](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/3_execution_model.md)
  - [Configuration flags](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/configuration_flags.md#list-of-available-flags)



## INTEL GRAPHICS(GPU)

- [Developer Documents for Intel Processor Graphics](https://software.intel.com/content/www/us/en/develop/articles/intel-graphics-developers-guides.html)
- [Intel® Graphics for Linux* - Programmer's Reference Manuals](https://www.intel.com/content/www/us/en/develop/documentation/intel-graphics-for-linux-programmers-reference-guide/top.html)
  - Intel® Data Center GPU Max Series
  - [2023 Intel® Arc™ A-Series Graphics and Intel Data Center GPU Flex Series(Alchemist/Arctic Sound-M Platform)](https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/alchemist-arctic-sound-m.html)



## Tools

- [Profiling Tools Interfaces for GPU](https://github.com/intel/pti-gpu)
- [Intel® Advisor User Guide](https://software.intel.com/content/www/us/en/develop/documentation/advisor-user-guide/top/design-for-gpu-offload/offload-modeling-perspective.html)
- [Intel VTune Profiler User Guide](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top.html)

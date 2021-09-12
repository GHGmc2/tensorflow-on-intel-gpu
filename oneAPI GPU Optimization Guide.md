# oneAPI GPU Optimization Guide

- [oneAPI GPU Optimization Guide](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-gpu-optimization-guide/top.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)



## Introduction

### Productive Performance not Performance Portability

Accelerator architectures are specialized, which includes **restructuring and tuning the code to create the best mapping of the application to the hardware**. The value of oneAPI is that it allows each of these variations to be expressed in a common language with device-specific variants launched on the appropriate accelerator.



### Phases in the Optimization Workflow

The first phase in using a GPU is to identify which characteristics of the code are most important when deciding what to offload. (The Intel Advisor tool)



GPUs often exploit parallelism at multiple levels. This includes overlap between host and GPU, parallelism across the compute cores, overlap between compute and memory accesses, concurrent pipelines, and vector computation.



*Keep all the compute resources busy.* Often you create many more independent tasks than available compute resources so that the hardware can schedule more work as prior tasks complete.



*Minimize the **synchronization** between the host and the device.* 

- minimize the number of times a kernel is launched.



*Minimize the **data transfer** between host and device.* 

- minimize data transfer by **keeping intermediate results on the device**
- **overlapping computation and data movement** so the compute cores never have to wait for data.



*Keep the data in faster memory and use an appropriate access pattern.*

- Registers, caches, and scratchpads are cheaper to access than local memory, but have smaller capacity.

- Use an access pattern that will use all the data before moving to the next chunk.
- Use a stride that avoids memory bank conflicts.



### Profiling and Tuning your Code

Illustrate how behavior compares with peak hardware roofline, and identify the most important hotspots to focus optimization effort.



## Getting Started

- Amdahl’s Law



### Locality Matters

An accelerator often has specialized memory **with a disjoint address space**.

Bringing data closer to the point of execution improves efficiency.

- Allocate your data on the accelerator, and when copied there keep it resident for as long as possible.
- Access contiguous blocks of memory as your kernel executes. The hardware will fetch contiguous blocks into the memory hierarchy, so you have already paid the cost for the entire block.
- Restructure your code into blocks with higher data reuse.



## Parallelization

three ways to develop parallel code

- Use a Parallel Programming Language or API: DPC++
- Parallelizing Compilers:  directive-based approach
- Parallel Libraries



## Intel UHD Graphics



### Execution Unit

Each EU is simultaneously multithreaded (SMT) with 7 threads. The primary computation units are *a pair of* **SIMD ALU**. Each ALU can execute up to four 32-bit floating-point or integer operations, or eight 16-bit floating-point operations.

![](https://software.intel.com/content/dam/dita/develop/oneapi-gpu-optimization-guide-0910-0955/_images/Gen-EU.PNG)

For convenience, we can model **an EU as executing 7 threads with 8 SIMD lanes** for a total of 56 concurrent 32-bit operations (or 112 concurrent 16-bit operations).

- Each **EU** can execute **eight** SIMD(width may be 8, 16 or 32) F32 vector operations (16 with FMA) and 16 SIMD F16 (32 with FMA).
- Each **hardware thread** has 128 general-purpose registers (GRF) of 32B wide(SIMD8) for the total of 4KB register/thread (or 28KB/EU). Each **GFR** can hold a vector of one, two, four, or eight 32-bit floating point or integer values or 16 16-bit values.



### SubSlice

Each SubSlice contains an EU array of 8 (Gen11) to 16 (Gen12) EUs. In addition to EUs, each SubSlice also contains:

- an **instruction cache**
- a **local thread dispatcher**
- a read-only **texture/image sampler** of 64B/cycle
- a **Dataport** of 64B/cycle for both read and write
- 64KB (capacity) banked shared local memory (**SLM**) accessible from the eight EUs.

![](https://software.intel.com/content/dam/dita/develop/oneapi-gpu-optimization-guide-0910-0955/_images/SubSlice.PNG)



For read-only data, it is possible to use the sampler unit to get an additional 64B/cycle data inputs into a SubSlice. The total read bandwidth across Dataport, Sampler and SLM is 192B/cycle.

One important usage of SLM is to **share global atomic data** among all the 448 (8EU * 7 Threads * 8 SIMD oprations) concurrent work-items executing in a SubSlice. For this reason, *if a kernel’s work-group contains synchronization operations, all work-items of the work-group must be allocated to a single SubSlice* so that they have shared access to the same 64KB SLM. In contrast, if a kernel does not access SLM, its work-items can be dispatched *across multiple SubSlices* for high occupancy and utilization.

The work-group size must be chosen carefully to maximize the occupancy and utilization of the SubSlice.



The **maximum number of work-groups** that can be executed on a single SubSlice is **16** [^5].



### Slice

Each Slice also contains a shared **L3 cache** and other slice-common units for graphics and media processing.



## DPC++ Thread Hierarchy and Mapping

- **Sub-group**: represents a short range of **consecutive work-items that are processed together as a SIMD vector** of length 8, 16, 32, or a multiple of the native vector length of a CPU with Intel UHD Graphics.

![](https://software.intel.com/content/dam/dita/develop/oneapi-gpu-optimization-guide-0910-0955/_images/Thread-Hierarchy-1.PNG)



### Thread Synchronization

Two synchronization mechanisms (only defined for work-items **within the same work-group**, DPC++ does not provide any synchronization mechanism inside a kernel across all work-items across the entire nd_range).

- `mem_fence`[^7] inserts a memory fence **on global and local memory** access across all work-items **in a work-group**.
- `barrier` **inserts a memory fence and blocks the execution of all work-items within the work-group until all work-items have reached its location**.



### Mapping Work-groups to SubSlices



#### example

```c++
auto command_group = [&](auto &cgh) {
    cgh.parallel_for(sycl::range<3>(64, 64, 64), // global range
                     [=](item<3> it) {
                         // (kernel code)
                     });
};
```



To fully utilize the 8 SIMD lanes, 8 EUs, and 7 thread contexts per EU, the compiler can pick a sub-group size of 8 and a group size of 448 (8 EU x 7 threads x 8 SIMD). The compiler can dispatch the groups over the eight SubSlices in the system[^10].



The two most important GPU resources are:

- **Thread Contexts**: The kernel should have a sufficient number of threads to utilize the GPU’s thread contexts.
- **SIMD Units and SIMD Registers**: The kernel should be organized to **vectorize the work-items** and **utilize the SIMD registers**.



Fully occupying a **SubSlice** requires 56 threads(**not work items**, 8EU * 7thread) in a work-group. Using more than 56 threads to the work-group will not increase hardware utilization. As a rule of thumb, we recommend 56 as the thread upper bound, 4 as the thread lower bound.



*Work group size = Threads x SIMD sub-group size*. The GPU compute driver may have additional constraints to the maximum work-group size.

DPC++ does not provide a mechanism to directly set the number of **threads** in a work-group. A programmer can explicitly specify sub-group size using `intel::reqd_sub_group_size({8|16|32})` to override the compiler’s selection.



We can increase the sub-group size **as long as there are a sufficient number of registers for the kernel** after widening. The effect of increasing sub-group size is similar to loop unrolling: while **each EU still executes eight 32-bit operations per cycle**, the amount of work per work-group interaction is doubled/quadruped. 



### Impact of Work-item Synchronization within Work-group

- example

```c++
auto command_group = [&](auto &cgh) {
    cgh.parallel_for(nd_range(sycl::range(64, 64, 64), // global range
                              sycl::range(1, R, 64)    // local range
                             ),
                     [=](sycl::nd_item<3> item) {
                         // (kernel code)
                         // Internal synchronization
                         item.barrier(access::fence_space::global);
                         // (kernel code)
                     });
};
```

**Synchronization is implemented using a SubSlice’s SLM** for shared variables, all the work-items of a work-group must be allocated to the same SubSlice, which affects SubSlice occupancy and kernel performance.



In the case of `R>7`, it requires more resources than a SubSlice can provide such that the kernel will **fail to launch**[^11]. In the case of `R=4`, a SubSlice is only 57% occupied (4/7) and *the three unused thread contexts are not sufficient to accept another work-group*, wasting 43% of the available EU capacities.



### Impact of Local Memory

Work-group local variables are shared among its work-items, they are allocated in a SubSlice’s SLM. Therefore, this work-group must be allocated to a single SubSlice.

In addition, one must also weigh the sizes of local variables such that the local variables fit within a SubSlice’s 64KB SLM capacity limit.



### A Detailed Example

In the absence of intra-work group synchronization, we know that threads from any work-group can be dispatched to any SubSlice.



## Kernels



### Reduction

There are several ways this can be parallelized, and care must be taken to ensure that the amount of communication/synchronization is minimized between different processing elements.



Examples:

- A naive way: use a global variable and let the threads to update this variable using an atomic operation.
- split the array into small chunks, let each thread compute a local sum for each chunk, and then do a sequential/tree reduction of the local sums.
- vectorize the loop.
- divide the work into enough work-groups to get full occupancy of all thread contexts. This allows the code to better tolerate long latency instructions.
- create a number of number of work-groups and do a tree reduction in each work-group(takes advantage of the very fast synchronization operations among the work-items in a work-group).
- multi-stage reduction
- built-in reduction operations
- use the shared local memory to store the intermediate results
- the blocking technique
- uses sub-group loads to generate the intermediate result in a vector form



Different implementations can have different performance characteristics which depends on the architecture of the accelerator. Another important thing to notice here is that time it takes to being the result of reduction to the host over the PCIE interface (for a discrete GPU) is almost same as actually doing the entire reduction on the device.



### Sub-groups

- examples: TODO



The mapping of work-items and work-groups to hardware execution units (EU) is implementation-dependent. 

Work-group execution may or or may not be preempted depending on the capabilities of underlying hardware. Work-items in the same work-group are guaranteed to run concurrently.



A sub-group is a collection of contiguous work-items that **execute in the same EU thread**. When the device compiler compiles the kernel, multiple work-items are packed into a sub-group by vectorization so the generated SIMD instruction stream can perform tasks of multiple work-items simultaneously.



By default, the compiler selects a sub-group size using device-specific information and a few heuristics. The user can override the compiler’s selection using a kernel attribute `intel::reqd_sub_group_size` to specify the **maximum** sub-group size. The valid sub-group sizes are device dependent(use `q.get_device().get_info<sycl::info::device::sub_group_sizes>()` to get the info).



#### Vectorization and Memory Access

Each EU is a multithreaded SIMD processor. The compiler generates SIMD instructions to pack multiple work-items in a sub-group to be executed simultaneously in an EU thread. The SIMD width (thus the sub-group size), selected by the compiler based on device characteristics and heuristics or requested explicitly by the kernel, can be 8, 16, or 32.



If one or more SIMD lanes (or kernel instances or work items) diverge, the thread executes **both** branch paths before the paths merge later, increasing dynamic instruction count. SIMD divergence negatively impacts performance.



example:

- Accessing contiguous memory in a work-item is often not optimal, initializing 16 contiguous integers in one SIMD instruction is more efficient.
- If work-items in a sub-group access a contiguous block of memory, we can use **the sub-group block access functions** to take advantage of **block load/store instructions**.



At the time of writing, block load/store does **not** work with sub-group size 32 on current Intel hardware. So the group size explicitly requested must be 16 or smaller.



#### Data Sharing

Because the work-items in a sub-group execute in the same thread, it is more efficient to share data between work-items. One way to share data among work-items in a sub-group is to use `shuffle` functions.



#### Work-item Private Variable Size, Register Spills, and Sub-group Size

To improve performance, the compiler promotes private variables in each work-item to registers. If not all private variables of all work-items in a sub-group can fit in the registers, variables that are not promoted to registers will spill into **global** memory.

By default, the compiler uses register pressure as one of the heuristics to choose SIMD width or sub-group size.



#### Sub-group Size vs. Maximum Sub-group Size

The sub-group size and maximum sub-group size are the same **if the work-group size is divisible by the maximum sub-group size and both sizes are powers of two**. The maximum sub-group size is actually the SIMD width so it does not change, but actual  sub-group size does.



### Shared Local Memory

Access to the SLM is limited to the work-items in the same work-group scheduled to execute on the EUs of the same SubSlice.

Because it is on-chip in each SubSlice, the SLM has much higher bandwidth and much lower latency than global memory.



It is often helpful to think of SLM as a work-group managed cache. The data stays in SLM during the lifetime of the work-group for faster access. After the work-group completes execution, the data in SLM is also gone and invalid.



#### Shared Local Memory Size and Work-group Size

The maximum SLM size can limit the work-group size.



#### Bank Conflicts

The SLM is divided into equal sized memory banks that can be accessed simultaneously for high bandwidth. At the time of writing, **64 consecutive bytes** are stored in **16 consecutive banks** at **4-byte granularity**. 

Requests to **different** banks can be serviced in parallel, but requests to the **same** bank (except **readings** from the **same address in the same bank**) have to be serialized, hence bank conflict.



#### Data Sharing and Work-group Barriers

When SLM data is shared, work-group barriers are often required for work-item synchronization. The barrier has a cost and the cost may increase with a larger work-group size.



#### Using SLM as Cache

We sometimes find it is more desirable to have the application manage caching(SLM) of some hot data than to have the hardware do it automatically(L3) for us.



### Removing Conditional Checks

If one or more work items take a divergent path, then both paths have to be executed before they later merge. In general, removing conditional checks can help performance.



examples:

- Padding Buffers to Remove Conditional Checks
- Replacing Conditional Checks with Relational Functions: `select()`, `min()`, `max()`, etc..



### Kernel Launch

- Kernel submission start time
- Kernel submission end time: The host performs multiple tasks like queuing the arguments, allocating resources in the runtime for the kernel to start execution on the device.
- Kernel Launch time: there are a few data transfers that need to happen before the actual kernel starts execution which is typically **not** accounted separately from kernel launch time.
- Kernel completion time: The current generation of devices are non-preemptive which means that once a kernel starts it has to complete its execution.



DPC++ built-in profiling API: `sycl::queue q{sycl::property::queue::enable_profiling()}`.



### Executing Multiple Kernels on the Device at the Same Time

Two kinds of queues:

- in-order queues: kernels are executed in the order they were submitted to the queue
- out-of-order queues: kernels can be executed in an arbitrary order (subject to the dependency constraints among them).

By default, when no property is specified the queue is out-of-order.



A heuristic the driver uses in deciding about the kernel submissions: when the driver sees the GPU is not active, it immediately submits the kernel for execution and then buffers the kernels in the queue.



It is also possible to **statically** partition a single device into sub-devices through the use of `create_sub_devices` function of `device class`. This provides more control to the programmer for submitting kernels to an appropriate sub-device. However, it does not have flexibility to move kernels from one sub-device to another.



At the time of writing, only the OpenCL backend is able to execute the kernels out of order.



### Synchronization among threads in a Kernel

- Atomic operations
- Fences: Fence primitives are used to order loads and stores. Fences can have acquire semantics, release semantics, or both.
- Barriers: used to synchronize sets of work-items within individual groups.
- Hierarchical parallel dispatch: synchronization within the work-group is made explicit through multiple instances of the `parallel_for_work_item` function call, rather than through the use of explicit work-group barrier operations.
- Device event



### Restrict Directive

When the compiler cannot determine whether pointers of inputs alias each other, it will conservatively assume that they do, in which case it will not reorder operations on these pointers.

It is possible to tell the compiler when it is safe to assume that pointers are not aliased(`intel::kernel_args_restrict`).

It is always better to issue the loads before any operations on them are done. This will allow the loads to complete before the data are actually needed for computation.



### Submitting Kernels to Multiple Queues

Queues hold a **context** that describes the state of the device. This state includes the contents of buffers and any memory needed to execute the kernels. The runtime keeps track of the current device context and avoids unnecessary memory transfers between host and device. Therefore, it is better to submit and launch kernels from one context together.



If the kernels are submitted to different queues that **share the same context**, the performance is similar to submitting it to one queue. The issue to note here is that when a kernel is submitted to a new queue with a different context, the JIT process compiles the kernel to the new device associated with the context.

This will prevent the need to transfer the input buffers, but the memory footprint of the kernels will increase because all the output buffers have be resident at the same time in the context whereas earlier the same memory on the device could be used for the output buffers. Another thing to remember is the issue of memory-to-compute ratio in the kernels. 



### Avoid Redundant Queue Construction

A context is constructed, either directly by the user or implicitly when creating a queue, to hold all the runtime information required by the SYCL runtime and the SYCL backend to operate on a device.

In general creating a new context is a heavy duty operation due to the need for JIT compiling the program every time a kernel is submitted to a queue with a new context.



### Considerations for selecting work-group size

In situations where there are no barriers or atomics used, the work-group size will not impact the performance. The threads created within a work-group and threads from different work-groups behave in a similar manner from scheduling and resourcing point of view when there are no barriers or shared memory in the work-groups.



## Memory

Utilizing the right level in the memory hierarchy is critical to get best performance.



### Performance Impact of USM and Buffers

#### USM

##### ~~Shared Allocations~~

Shared allocations only transfer the data that the host actually references, with a memory page granularity.



##### Device Allocations

With device allocation, data can only be directly accessed on the device and must be explicitly copied to the host. All synchronization between device and host are explicit.

Device allocations can provide the best performance.



Best practices:

- Optimizing Memory Movement between Host and Accelerator
- Avoid moving data back and forth between host and device
- Avoid Declaring Buffers in a Loop
- Use Appropriate access Mode on Accessors to Buffers



## Host/Device Coordination

### Asynchronous and Overlapping Data Transfers between Host and Device

#### Bandwidth between Host and Accelerator

The local memory bandwidth of an accelerator is an order of magnitude higher than host-to-device bandwidth over a link like PCIe.



#### Overlapping Data Transfer from Host to Device with Computation on Device

Some GPUs provide specialized engines for copying data from host to device. In systems where there is a copy engine that can be used to transfer data between host and device, we can see that the kernels from different loop iterations can execute in parallel. The parallel execution can manifest in two ways:

- Between two memory copies, where one is executed by the GPU EUs and one by the copy engine
- Between a memory copy and a compute kernel, where the memory copy is executed by the copy engine and the compute kernel by the GPU EUs.



## Using multiple heterogeneous devices

DPC++ provides the ability to treat the CPUs and the accelerators uniformly to distribute work among them. It is the responsibility of the programmer to ensure a balanced distribution of work among the heterogeneous compute resources in the platform.

The host CPU can be treated as an accelerator and the DPCPP can submit kernels to it for execution. This is completely independent and orthogonal to the job done by the host to orchestrate the kernel submission and creation. The underlying **operating system** manages the kernels submitted to the CPU accelerator as another process and uses the same openCL/Level0 runtime mechanisms to exchange information with the host device.



## Compilation

For the kernels, this might be ahead-of-time (AOT) or just-in-time (JIT).



### Just-In-Time Compilation in DPC++

The Intel® oneAPI DPC++ Compiler converts a DPC++ program into **an intermediate language called SPIR-V** and stores that in the binary produced by the compilation process. The **advantage** of producing this intermediate file instead of the binary is that this code can be run on any hardware platform by **translating the SPIR-V code into the assembly code of the platform at runtime**. This process of translating the intermediate code present in the binary is called JIT compilation (just-in-time compilation). JIT compilation can happen on demand at runtime. 



If the application contains multiple kernels, one can force eager JIT compilation or lazy JIT compilation using compile-time switches. **Eager** JIT compilation will invoke the JITter on all the kernels in the binary at the beginning of execution, while **lazy** JIT compilation will enable the JITter only when the kernel is actually called during execution.

This mode can be enabled during compilation using the following option:

```bash
-fsycl-device-code-split=<value>
```

- *per_kernel*: generates code to do JIT compilation of a kernel only when it is called
- *per_source*: generates code to do JIT compilation of **all** kernels in the source file when **any** of the kernels in the source file are called
- *off*: the default, which does eager JIT compilation of all kernels in the application



### Specialization Constants

DPC++ has a feature called *specialization constants* that can **explicitly trigger JIT compilation** to generate code from the intermediate SPIR-V code based on the **runtime** values of these specialization constants. These JIT compilation actions are done during the execution of the program **when the values of these constants are known**.



## Debugging and Profiling

Understanding the behavior of your system is critical to making informed decisions about optimization choices.



### [pti-gpu](https://github.com/intel/pti-gpu)

- [Level Zero Tracer](https://github.com/intel/pti-gpu/tree/master/tools/ze_tracer)
- [System Monitoring Utility](https://github.com/intel/pti-gpu/tree/master/tools/sysmon)



### [VTune](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top.html)

- GPU Offload Analysis (technical preview)
- GPU Compute/Media Hotspots Analysis (technical preview)
  - Source Analysis



### [Advisor]((https://software.intel.com/content/www/us/en/develop/documentation/advisor-user-guide/top.html))

- Offload Modeling: produces upper-bound speedup estimates using a bounds-and-bottlenecks performance model.
- GPU Roofline Insights



### Doing IO in the kernel

In accelerators, printing is surprisingly hard and also fairly expensive in terms of overhead.

File I/O is not possible from DPC++ kernels.



The *stream* class provides functionality that is very similar to the C++ STL *ostream* class, and its usage is similar to the STL class.

```c++
stream(size_t BufferSize, size_t MaxStatementSize, handler &CGH);
```

Care must be taken in choosing the appropriate *BufferSize* and *MaxStatementSize* parameters. Sizes that are insufficient may cause statements to either not be printed, or to be printed with less information than expected.



The message or data that needs to be printed is sent to the SYCL *stream* instance via the appropriate *operator<<* method. SYCL provides implementations for all the builtin data types (such as *int*, *char* and *float*) as well as some common classes (such as *sycl::nd_range* and *sycl::group*).

The output from `sycl::stream` is printed after the kernel has completed execution.



### Using the timers

DPC++ provides a profiling capability which allows one to keep track of the time it took to execute kernels.

```c++
return (event.template get_profiling_info<info::event_profiling::command_end>() -
       event.template get_profiling_info<info::event_profiling::command_start>());
```



The DPC++ profiling does **not include any data transfer times** between the host and the offload device.



# Comments

[^5]: only if use SLM?
[^7]: barrier vs. memory fence
[^10]: [Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) (ND-range Stride Loops)?
[^11]: can not be validated

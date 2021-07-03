Level Zero

- [spec](https://spec.oneapi.com/level-zero/latest/index.html)
- [Header files](https://github.com/oneapi-src/level-zero)



# Introduction



## Objective

- Core: the lowest-level, fine-grain and most explicit control
- Tools: low-level access to device capabilities
- System Management



## Fundamentals

- [Error Handling](https://spec.oneapi.com/level-zero/latest/core/INTRO.html#error-handling)
- [Multithreading and Concurrency](https://spec.oneapi.com/level-zero/latest/core/INTRO.html#multithreading-and-concurrency): the API is designed to be free-threaded rather than thread-safe.
  - The exception to this rule is that all memory allocation APIs are thread-safe since they allocate from a single global memory pool.
- ABI: An ABI in this context means the size, alignment, and layout of C data types; the procedure calling convention; and the naming convention for shared library symbols corresponding to C functions.
  - Device-Driver Interfaces (DDIs)



### Multithreading and Concurrency

The primary usage-model enabled by these rules is:

- multiple, simultaneous threads may operate on independent driver objects with no implicit thread-locks
- driver object handles may be passed between and used by multiple threads with no implicit thread-locks



# Core



## Drivers and Devices

The device, sub-device and memory are exposed at **physical** level while command queues, events and synchronization methods are defined as **logical** entities.



### Drivers

A driver object represents a collection of physical devices in the system accessed by the same Level-Zero driver. More than one driver may be available in the system.



### Device

A device object represents a physical device in the system that supports Level-Zero.



### Initialization and Discovery

The Level-Zero API must be initialized by calling `zeInit` before calling any other API function. This function will load all Level-Zero driver(s) in the system into memory for the current process, for use by all Host threads.



## Contexts

A context is **a logical object used by the driver** for managing all memory, command queues/lists, modules, synchronization objects, etc.

- A context handle is primarily used during **creation and management of resources that may be used by multiple devices**.



An application may optionally create multiple contexts

- The primary usage-model for multiple contexts is **isolation of memory and objects** for multiple libraries within the same process.
- The same context may be used simultaneously on multiple Host threads.



If a device was hung or reset, then the context is no longer valid.

In order to recover, the context must be destroyed. After the device is reset, the application can create a new context and continue operation.



## Memory

For GPUs, the API exposes two levels of the device memory hierarchy:

1. Local Device Memory: can be managed at the device and/or sub device level.
2. Device Cache(s):
   - Last Level Cache (L3) can be controlled through memory allocation APIs.
   - Low Level Cache (L1) can be controlled through program language intrinsics.



### Memory

#### Types

The type of allocation describes the ownership of the allocation:

- Host
  - accessible by the host and one or more devices.
  - trade off wide accessibility and transfer benefits for potentially higher per-access costs
- Device
  - generally trade off access limitations for higher performance.
- Shared
  - trade off transfer costs for per-access benefits.



A **Shared System** allocation is a sub-class of a **Shared** allocation, where the memory is allocated by a *system allocator* (such as `malloc` or `new`) rather than by an allocation API.



At a minimum, drivers will assign unique physical pages for each device and shared memory allocation.



### Reserved Device Allocations

An application can reserve a range of the virtual address space and map this to physical memory as needed. This provides flexibility for applications to manage large dynamic data structures which can grow and shrink over time while maintaining optimal physical memory usage.



## Command Queues and Command Lists

The motivations for separating a command queue from a command list:

- Command **queues** are mostly **associated with physical device properties** (such as the number of input streams), provide near zero-latency access to the device.
- Command **lists** are mostly **associated with Host threads** for simultaneous construction (independently of command queue submission).



```mermaid
graph LR
A(physical device engines) --- C("command queue group (physical input stream)") --- E("command queue (logical input stream)") --- F(command list)
```



![](https://spec.oneapi.com/level-zero/latest/_images/core_queue.png)



### Command Queue Groups

A command queue group represents **a physical input stream, which represents one or more physical device engines**.



#### Discovery

- The number of physical engines within a group
- The types of commands supported by the group



### Command Queues

A command queue represents **a logical input stream to the device, tied to a physical input stream**.



#### [Create](https://spec.oneapi.com/level-zero/latest/core/PROG.html#creation)

- At creation time, the command queue is explicitly **bound to a command queue group** via its ordinal.
- Multiple command queues created for the same command queue group on the same context, may also share the same physical hardware context.
- All command lists executed on a command queue are guaranteed to **only** execute on an engine from the command queue group which it is assigned.



#### [Execution](https://spec.oneapi.com/level-zero/latest/core/PROG.html#execution)

- Command lists **submitted** to a command queue are **immediately submitted to the device** for execution.
- Submitting multiple commands lists in a single submission allows an implementation the opportunity to optimize across command lists.
- Command queue submission is free-treaded, allowing multiple *Host threads*(???) to share the same command queue. If multiple Host threads enter the same command queue simultaneously, then execution order is undefined.
- Command lists can only be executed **on a command queue with an identical command queue group ordinal**.
- If a device contains multiple sub-devices, then command lists submitted to a **device-level command queue** may be optimized by the driver to fully exploit the concurrency of the sub-devices by distributing command lists across sub-devices.



#### Destruction

- The application is responsible for making sure the device is not currently executing from a command queue before it is deleted. This is typically done by tracking command queue fences, but may also be handled by calling `zeCommandQueueSynchronize`.



### Command Lists

A command list represents a sequence of commands for execution on a command queue.



#### Creation

- A command list is created for **execution on a specific type of command queue, specified using the command queue group ordinal**.



#### [Appending](https://spec.oneapi.com/level-zero/latest/core/PROG.html#appending)

- An application may share a command list handle across multiple Host threads. However, the application is responsible for **ensuring that multiple Host threads do not access the same command list simultaneously**.
- **By default, commands are started in the same order in which they are appended**. However, an application may allow the driver to optimize the ordering, reordering is guaranteed to only occur between barriers and synchronization primitives.
- By default, commands submitted to a command list are optimized for execution by **balancing both device throughput and Host latency**.
- If a device contains multiple sub-devices, then commands submitted to a *device-level command list*(???) may be optimized by the driver to fully exploit the concurrency of the sub-devices by distributing commands across sub-devices.



#### Submission

- A command list may be submitted to any or multiple command queues.
- By definition, **a command list cannot be executed concurrently on multiple command queues**.
- Command lists do not inherit state from other command lists executed on the same command queue.
- A command list may be submitted multiple times. It is up to the application to ensure that the command list can be executed multiple times.



#### Recycling

- The application is responsible for making sure the device is not currently executing from a command list before it is reset or deleted.



#### Low-Latency Immediate Command Lists

Used for very low-latency submission usage-models.

- is **both a command list and an implicit command queue**, and is created using a command queue descriptor.
- Commands submitted to an immediate command list are immediately executed on the device.
- is not required to be closed or reset.



## Synchronization Primitives

There are two types of synchronization primitives:

- Fence: is heavyweight and used to communicate to the host that command queue execution has completed.
- Events: used as fine-grain host-to-device, device-to-host or device-to-device execution and memory dependencies.



The motivations for separating the different types of synchronization primitives:

- Allows device-specific optimizations for certain types of primitives
  - Fences may share device memory with all other fences within the same command queue.
  - Events may be implemented using pipelined operations as part of the program execution.
  - Fences are implicit, coarse-grain execution and memory barriers. Events optionally cause fine-grain execution and memory barriers.
- Allows distinction on which type of primitive may be shared across devices.



Events are generic synchronization primitives that can be used across many different usage-models, including those of fences. However, this generality comes with some cost in memory overhead and efficiency.



### Fences

- is **associated with a single command queue**.
- can only be signaled from a device’s command queue and can only be waited upon from the host.
- **guarantees both execution completion and memory coherency, across the device and host**, prior to being signaled.
- has only two states: not signaled and signaled.
- A fence doesn’t implicitly reset.
- can only be reset from the Host.
- cannot be shared across processes.



The primary usage model for fences is to notify the Host when a command list has finished execution.

![](https://spec.oneapi.com/level-zero/latest/_images/core_fence.png)



### Events

- can be
  - Signaled from within a device’s command list and waited upon within the same command list
  - Signaled from within a device’s command list and waited upon from the host, another command queue or another device
  - Signaled from the host, and waited upon from within a device’s command list.
- has only two states: not signaled and signaled.
- An event doesn’t implicitly reset, can be explicitly reset from the Host or device.
- can be appended into multiple command lists simultaneously.
- can be shared across devices and processes.
- can invoke an execution and/or memory barrier; which should be used sparingly to avoid device underutilization.
- There are no protections against events causing deadlocks, these problems are left to the application to avoid.
- An event intended to be signaled by the host, another command queue or another device after command list submission to a command queue may prevent subsequent forward progress within the command queue itself. This can create bubbles in the pipeline or deadlock situations if not correctly scheduled.




#### Kernel Timestamp Events

is a special type of event that records device timestamps at the start and end of the execution of kernels, to provide a duration of execution.



## Barriers

There are two types of barriers:

- Execution Barriers - used to communicate **execution dependencies** between commands within a command list or across command queues, devices and/or Host.
- Memory Barriers - used to communicate **memory coherency dependencies** between commands within a command list or across command queues, devices and/or Host.



#### Execution Barriers

Commands executed on a command list are only guaranteed to **start** in the same order in which they are submitted; i.e. there is no implicit definition of the order of **completion**.

- Fences: indicate that all previous commands must complete prior to the fence being signaled.
- Events provide explicit, fine-grain control over execution dependencies between commands; allowing more opportunities for concurrent execution and higher device utilization.



#### Memory Barriers

Commands executed on a command list are not guaranteed to maintain memory coherency with other commands.

- Fences: indicate that all caches and memory are coherent across the device and Host prior to the fence being signaled.
- Events provide explicit, fine-grain control over cache and memory coherency dependencies between commands; allowing more opportunities for concurrent execution and higher device utilization.



#### Range-based Memory Barriers

provide explicit control of which cachelines require coherency.



## Modules and Kernels

- Modules represent a single translation unit that consists of kernels that have been compiled together.
- Kernels represent the kernel within the module that will be launched directly from a command list.

![](https://spec.oneapi.com/level-zero/latest/_images/core_module.png)



## Advanced



### Sub-Device Support

There are functions to query and obtain sub-devices, but outside of these functions there are no distinctions between sub-devices and devices.



### Inter-Process Communication

Since each process has its own virtual address space, there is no guarantee that the same virtual address will be available when the memory object is shared in new process.



There are two types of Inter-Process Communication (IPC) APIs for using Level-Zero allocations across processes:

- Memory
- Events



### P2P Access and Queries

Peer to Peer API’s provide capabilities to marshall data across Host to Device, Device to Host and Device to Device.

Data coherency is maintained by the driver without any explicit involvement from the application.



# Tools

## [Metrics](https://spec.oneapi.com/level-zero/latest/tools/PROG.html#metrics)





# Sysman





# Extensions



## List of Standard Extensions
- “ZE_extension_cache_reservation”

- “ZE_extension_float_atomics”

- “ZE_extension_linkonce_odr”

- “ZE_extension_raytracing”

- “ZE_extension_subgroups”



## List of Experimental Extensions
- “ZE_experimental_event_query_timestamps”
- “ZE_experimental_global_offset”
- “ZE_experimental_scheduling_hints”
- “ZE_experimental_module_program”
- “ZE_experimental_relaxed_allocation_limits”
- “ZET_experimental_api_tracing”





# [API Doc](https://spec.oneapi.com/level-zero/latest/api.html)

## [Core API](https://spec.oneapi.com/level-zero/latest/core/api.html)




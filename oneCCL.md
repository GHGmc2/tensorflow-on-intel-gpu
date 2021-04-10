oneCCL

 - [doc](https://oneapi-src.github.io/oneCCL/)



# Developer Guide

### PROGRAMMING MODEL



#### Concepts

- Environment
- Stream
- Communicator



#### Collective Communication



##### Collective Opertations

- Allgatherv
- Allreduce
- Reduce
- Alltoall
- Barrier
- Broadcast



### GENERAL CONFIGURATION



### ADVANCED CONFIGURATION



#### [Collective algorithms](https://github.intel.com/pages/ict/mlsl2/env_variables.html#available-algorithms)



#### Caching



#### [Prioritization](https://github.intel.com/pages/ict/mlsl2/env_variables.html#ccl-priority)

- None - default mode when all collective operations have the same priority.
- Direct - priority is explicitly specified by users using  `coll_attr.priority`.
- LIFO (Last In, First Out) - priority is implicitly increased on each collective call. In this case user doesn’t have to specify priority.



#### [Fusion](https://github.intel.com/pages/ict/mlsl2/env_variables.html#ccl-fusion)

- CCL_FUSION_BYTES_THRESHOLD
- CCL_FUSION_COUNT_THRESHOLD
- CCL_FUSION_CYCLE_MS



#### Sparse collective operations



#### Unordered collectives support



#### Fault tolerance / elasticity



### REFERENCE MATERIALS



#### Environment variables



#### [Library API](https://github.intel.com/pages/ict/mlsl2/api/library_root.html)

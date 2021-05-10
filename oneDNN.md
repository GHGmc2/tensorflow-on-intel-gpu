oneDNN

# oneDNN

- [latest doc](https://oneapi-src.github.io/oneDNN/)



## Programming Model



### [Basic Concepts](https://docs.oneapi.com/versions/latest/onednn/dev_guide_basic_concepts.html)

In essence, the oneDNN programming model consists in **executing one or several primitives to process data in one or several memory objects. The execution is performed on an engine in the context of a stream**.

![](https://oneapi-src.github.io/oneDNN/img_programming_model.png)



**Primitives**

A *primitive* is a functor object that encapsulates a particular computation. Additionally, using primitive *attributes* certain primitives can represent more complex **fused** computations.

The most important difference between a primitive and a pure function is that a primitive can store state. 

- One part of the primitive’s state is **immutable**. This approach allows oneDNN primitives to pre-generate code specifically tailored for the operation to be performed. 
- The **mutable** part of the primitive’s state is referred to as a scratchpad. It is a memory buffer that a primitive may use for temporary storage only during computations. The scratchpad can either be owned by a primitive object (which makes that object non-thread safe) or be an execution-time parameter.



**Engines** is an abstraction of a computational device.

**Streams** encapsulate execution context tied to a particular engine.

**Memory objects** encapsulate handles to memory allocated on a specific engine.


### Primitive Attributes

#### [Post-ops](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_post_ops.html)

 - Eltwise
 - Sum
 - Depthwise
 - Binary

Different post-ops can be chained together by appending one after another. Note that the appending order matters: the sequence of the post operations is executed in the order of appearance. The maximum number of post operations supported in the library is 32.
Moreover, the support might also depend on the actual implementation of a primitive. For instance, the library may not support post-ops for primitive reference implementations.



### [Interoperability with DPC++ and OpenCL](https://docs.oneapi.com/versions/latest/onednn/usergroup2.html)



#### [DPC++ Interoperability](https://docs.oneapi.com/versions/latest/onednn/dev_guide_dpcpp_interoperability.html)

- The mapping between oneDNN and SYCL objects is provided in the following table:

| oneDNN object | SYCL object(s)                             |
| :------------ | :----------------------------------------- |
| Engine        | `cl::sycl::device` and `cl::sycl::context` |
| Stream        | `cl::sycl::queue`                          |
| Memory        | `cl::sycl::buffer<uint8_t, 1>`             |



- API to Construct oneDNN Objects

| oneDNN object | API to construct oneDNN object                               |
| :------------ | :----------------------------------------------------------- |
| Engine        | [dnnl::engine(kind, sycl_dev, sycl_ctx)](https://docs.oneapi.com/versions/latest/onednn/structdnnl_1_1engine.html) |
| Stream        | [dnnl::stream(engine, sycl_queue)](https://docs.oneapi.com/versions/latest/onednn/structdnnl_1_1stream.html) |
| Memory        | [dnnl::memory(memory_desc, engine, sycl_buf)](https://docs.oneapi.com/versions/latest/onednn/structdnnl_1_1memory.html) |

- API to Access SYCL Objects

| oneDNN object | API to access SYCL object(s)                                 |
| :------------ | :----------------------------------------------------------- |
| Engine        | [dnnl::engine::get_sycl_device()](https://docs.oneapi.com/versions/latest/onednn/structdnnl_1_1engine.html#a8fff4f387e9c975166606a0182b37fbf) and [dnnl::engine::get_sycl_context()](https://docs.oneapi.com/versions/latest/onednn/structdnnl_1_1engine.html#a362fb3ae5876f2b3be1cf2044c611472) |
| Stream        | [dnnl::stream::get_sycl_queue()](https://docs.oneapi.com/versions/latest/onednn/structdnnl_1_1stream.html#a37cdaf0769013debc1dde3559f00cecc) |
| Memory        | [dnnl::memory::get_sycl_buffer()](https://docs.oneapi.com/versions/latest/onednn/structdnnl_1_1memory.html#a170571437afeabe2923477fdd6eea2fd) |



## Advanced Topics



### [Understanding Memory Formats](https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html)



#### Plain data formats

- from inner-most to outer-most

![](https://docs.oneapi.com/versions/latest/onednn/mem_fmt_img2.png)

- **NHWC**: `offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c`

  In this case the inner-most dimension is channels (`[b:0]`) that is followed by width (`[b:1]`), height (`[b:2]`), and finally batch (`[b:3]`).

- **NCHW**: `offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w`



#### Blocked layout

In order to achieve **better vectorization and cache reuse**, oneDNN introduces blocked layout that splits one or several dimensions into the **blocks of fixed size**. The most popular oneDNN data format is **nChw16c** on AVX512+ systems and **nChw8c** on SSE4.1+ systems.

 The offset function for **nChw8c** is:

```
offset_nChw8c(n, c, h, w) = n * CHW
                          + (c / 8) * HW*8
                          + h * W*8
                          + w * 8
                          + (c % 8)
```

![](https://docs.oneapi.com/versions/latest/onednn/mem_fmt_blk.png)

Note that blocks of 8 channels are kept contiguously in memory. Pixel by pixel the spatial domain is covered. Then next **slice** covers the subsequent 8 channels. Once all channel blocks are covered, the next **image** in the batch appears.



Zero-padding: to round the channels up to make them multiples of the block size and pad the resulting tail with zeros. (The actual size should always be queried via [dnnl::memory::desc::get_size()](https://docs.oneapi.com/versions/latest/onednn/structdnnl_1_1memory_1_1desc.html#ac20108bc192912382aa4a95ae27df804) in C++)




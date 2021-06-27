# [oneMKL DPC++ Developer Reference](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-mkl-dpcpp-developer-reference/top.html)



This publication describes the Data Parallel C++ (DPC++) interface.

- Basic Linear Algebra Subprograms (BLAS): vector, matrix-vector, and matrix-matrix operations.
- Sparse BLAS: basic operations on sparse vectors and matrices.
- LAPACK: solve systems of linear equations, least square problems, eigenvalue and singular value problems, and Sylvester’s equations.
- Random Number Generators: a set of routines implementing commonly used pseudorandom and quasi-random generators with continuous and discrete distributions.
- Summary Statistics: provides routines that compute basic statistical estimates for single and double precision multi-dimensional datasets.
- Vector Mathematics Functions: compute core mathematical functions on vector arguments.
- Fourier Transform Functions: offer several options for computing Fast Fourier Transforms (FFTs).



## Introduction to oneMKL BLAS and LAPACK with DPC++

In oneMKL, all DPC++ routines and associated data types belong to the oneapi::mkl namespace.



By default **column major** layout is assumed for all BLAS functions in oneapi::mkl::blas namespace, to use row major layout to store matrices, BLAS functions in oneapi::mkl::blas::row_major namespace must be used.

Currently LAPACK DPC++ APIs do not support matrices stored using row major layout.



### Differences between Standard BLAS/LAPACK and DPC++ oneMKL APIs

- All DPC++ objects (buffers and queues) are passed by **reference**, rather than by pointer. Other parameters(Scalar) are typically passed by value.
- DPC++ uses **buffers** to store data on a device and share data between devices and the host. Currently, all buffers must be **one-dimensional**.
- Row major layout is not supported directly, but you may still use DPC++ BLAS by treating row major matrices as transposed column major matrices.



For brevity, the `cl::sycl` namespace is omitted from DPC++ object types

A question mark (?) in a routine name stands for one or more characters (typically one of s, d, c, z) specifying the precision of the operation.



## Data Types

### BLAS and LAPACK Data Types

- transpose
- uplo: specifies whether the lower or upper triangle of a triangular, symmetric, or Hermitian matrix should be accessed.
- diag
- side
- offset



### Vector Math Data Types

Slices are used in the DPC++ VM Strided APIs. oneMKL slices accept positive, zero, and negative strides, for forward, static, and backward traversals of a vector, respectively.



## [Matrix Storage](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-mkl-dpcpp-developer-reference/top/matrix-storage.html)

- General Matrix
- Triangular Matrix
- Band Matrix
- Triangular Band Matrix

- Packed Triangular Matrix
- Vector



## Error Handling

Should errors occur, they are propagated at the point of a function call where they are caught using standard C++ error handling mechanisms.



## BLAS and Sparse BLAS Routines

- Level 1: vector-vector operations
- Level 2: matrix-vector operations
- Level 3: matrix-matrix operations
- 



## LAPACK Routines





## Fourier Transform Functions

The DPC++ interface computes an FFT in four steps:

1. Allocate a fresh descriptor for the problem with a call to the descriptor object constructor
2. Optionally adjust the descriptor configuration with a call to the descriptor<PRECISION, DOMAIN>::set_value function as needed.
3. Commit the descriptor with a call to the descriptor<PRECISION, DOMAIN>::commit function, that is, make the descriptor ready for the transform computation. Once the descriptor is committed, the parameters of the transform are “frozen” in the descriptor.
4. Compute the transform with a call to the `compute_forward` or `compute_backward` functions as many times as needed.




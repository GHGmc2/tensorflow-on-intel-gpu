eigen

# eigen

- `unsupported/Eigen/CXX11/src/Tensor`:

  - `TensorBase.h`: all kinds of functions...

    - nullaryExpr()

    - unaryExpr()

      ```c++
      template <typename CustomUnaryOp>
      const TensorCwiseUnaryOp<CustomUnaryOp, const Derived>
      unaryExpr(const CustomUnaryOp& func) const {
        return TensorCwiseUnaryOp<CustomUnaryOp, const Derived>(derived(), func);
      }
      ```

    - binaryExpr()

    - eval()

      ```c++
      const TensorForcedEvalOp<const Derived>
      eval() const {
        return TensorForcedEvalOp<const Derived>(derived());
      }
      ```

  - `TensorMap.h`

  - `TensorExpr.h`: `class TensorCwise(Nullary/Unary/Binary/Ternary)Op`

  - `TensorEvaluator.h`: responsible for the evaluation of the tensor expression. 特化分散在各个文件里。。

    ```c++
    template<typename NullaryOp, typename ArgType, typename Device>
    struct TensorEvaluator<const TensorCwiseNullaryOp<NullaryOp, ArgType>, Device> {
      // ...
    };
    
    // UnaryOp, BinaryOp, TernaryOp, SelectOp...
    ```

    

- submit kernel in Eigen: `TensorExecutor.h`, `TensorEvaluator.h`, `TensorDeviceGpu.h`

  ```mermaid
  graph TD
  
  A(TensorExecutor< Expression>::run) -- "TensorEvaluator(Expression, GpuDevice)" --> B(TensorEvaluator< TensorCwiseNullaryOp>.evalSubExprsIfNeeded)
  B -- if True --> C(GpuDevice.nullary_kernel_launcher)
  C --> D("stream->submit(range, kernel)")
  ```

- call stack example:

  - `TensorExecutor.h`: launch the evaluation of the expression on the specified computing device.

    ```c++
    // kernel functor
    template <typename Evaluator>
    struct ExecExprFunctorKernel {
      template <typename Scratch>
      ExecExprFunctorKernel(const Scratch, Evaluator evaluator_, const int range_) 
        : evaluator(evaluator_), range(range_) {}
      
      void operator()(cl::sycl::nd_item<1> itemID) {
       compute(itemID);
      }
      
      template <bool is_vec = Evaluator::PacketAccess>
      EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename std::enable_if<!is_vec>::type
      compute(const cl::sycl::nd_item<1>& itemID) {
        Index gId = static_cast<Index>(itemID.get_global_linear_id());
        Index total_threads = itemID.get_global_range(0);
    
        for (Index i = gId; i < range; i += total_threads) {
          evaluator.evalScalar(i);
        }
      }
      
      template <bool is_vec = Evaluator::PacketAccess>
      EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE typename std::enable_if<is_vec>::type
      compute(const cl::sycl::nd_item<1>& itemID) {
        const Index vectorizedRange = (range / Evaluator::PacketSize) * Evaluator::PacketSize;
        Index gId = static_cast<Index>(itemID.get_global_linear_id());
        const Index step = Evaluator::PacketSize * itemID.get_global_range(0);
        const Index start = Evaluator::PacketSize * gId;
        for (Index i = start; i < vectorizedRange; i += step) {
          evaluator.evalPacket(i);
        }
        gId += vectorizedRange;
        for (Index i = gId; i < range; i += itemID.get_global_range(0)) {
          evaluator.evalScalar(i);
        }
      }
    };
    
    // TensorExecutor::run()
    template <typename Expression, bool Vectorizable, TiledEvaluation Tiling>
    void TensorExecutor<Expression, GpuDevice, Vectorizable, Tiling>::run(const Expression& expr, const GpuDevice& dev) {
      typedef Eigen::TensorEvaluator<Expression, GpuDevice> Evaluator;
      Evaluator evaluator(expr, dev);
      const bool needs_assign = evaluator.evalSubExprsIfNeeded(NULL);
      if (needs_assign) {
        Index range, GRange, tileSize;
        Index total_size = ::Eigen::internal::array_prod(evaluator.dimensions());
        total_size = (total_size == 0) ? 1 : total_size;
        const int PacketSize = Eigen::PacketType<typename Evaluator::CoeffReturnType, GpuDevice>::size;
        Index vectorizable_threads = static_cast<Index>(total_size / PacketSize);
        dev.parallel_for_setup(vectorizable_threads, tileSize, range, GRange);
        range = total_size;
    
        dev.template nullary_kernel_launcher<typename Evaluator::CoeffReturnType, ExecExprFunctorKernel<Evaluator> >(
            evaluator,
            cl::sycl::nd_range<1>(cl::sycl::range<1>(GRange), cl::sycl::range<1>(tileSize)),
            Index(1),
            range);
      }
      evaluator.cleanup();
    }
    ```

    

  - `TensorDeviceGpu.h`

    ```c++
    template <typename OutScalar, typename sycl_kernel, typename InPtr, typename OutPtr, typename Range, typename Index, typename... T>
    void unary_kernel_launcher(const InPtr &inptr, OutPtr &outptr, Range thread_range, Index scratchSize, T... var) const {
      auto kernel_functor = [=](cl::sycl::handler &cgh) {
        typedef cl::sycl::accessor<OutScalar, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> LocalAccessor;
        LocalAccessor scratch(cl::sycl::range<1>(scratchSize), cgh);
        // init kernel functor with params
        cgh.parallel_for(thread_range, sycl_kernel(scratch, inptr, outptr, var...));
      };
      cl::sycl::event e;
      EIGEN_SYCL_TRY_CATCH(e = stream_->stream()->submit(kernel_functor));
      async_synchronize(e);
    }
    ```






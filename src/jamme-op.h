
#ifndef TENSORFLOW_KERNELS_JAMME_OP_H_
#define TENSORFLOW_KERNELS_JAMME_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct JamMeFunctor {
  void operator()(const OpKernelContext* ctx, const int size, const T* in, T* jam, T* out, float *sim);
};

}	// namespace functor
}	// namespace tensorflow

#endif /* TENSORFLOW_KERNELS_JAMME_OP_H_ */


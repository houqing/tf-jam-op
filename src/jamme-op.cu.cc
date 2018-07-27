
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "jamme-op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "cuda/include/cuda_fp16.h"


#define HALF_EXPONENT_BIT_MASK	(0x7c00)
#define HALF_MANTISSA_BIT_MASK	(0x03ff)

// 
#define JAM_BIT_NUM	0x2

// TODO
#define JAM_BIT_MASK	0x3
#define JAM_FLAG_MASK	0x8
#define JAM_BIT_RATIO	100

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

__global__ void JamMeRandCudaKernel(const int nthreads, const Eigen::half* in, Eigen::half* jam, Eigen::half* out, float *sim) {
	CUDA_1D_KERNEL_LOOP(idx, nthreads) {
		unsigned short tmp;
		unsigned short mask = (1<<JAM_BIT_NUM) -1;

		if ((((__half_as_ushort(in[idx])) << 1) >> 11) == 0x1f) { //NaN or Infinit
			jam[idx] = in[idx];
		} else if (in[idx] == (__ushort_as_half(0))) {
			jam[idx] = in[idx];
		} else if ((__half_as_ushort(out[idx])%100) < JAM_BIT_RATIO) {
			tmp = ((__half_as_ushort(in[idx])) & (~mask)) | ((__half_as_ushort(out[idx])) & mask);
			jam[idx] = __ushort_as_half(tmp);
		} else {
			jam[idx] = in[idx];
		}
	}

	sim[0] = 11;
}

__global__ void JamMeRandCudaKernel(const int nthreads, const float* in, float* jam, float* out, float *sim) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
	  jam[idx] = in[idx];
  }

  sim[0] = 12;
}

#if 0
template <typename T>
__global__ void JamMeRandCudaKernel_TODO(const int nthreads, const T* in, T* jam, T* out, float *sim) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    jam[idx] = in[idx];
    if (((__half_as_ushort(in[idx]) & HALF_EXPONENT_BIT_MASK) != 0) && ((__half_as_ushort(out[idx]) % 100) < JAM_BIT_RATIO)) {
	    unsigned short mi;
	    unsigned short jv;
	    mi = __half_as_ushort(in[idx]) & HALF_MANTISSA_BIT_MASK;
	    jv = __half_as_ushort(out[idx]) & JAM_BIT_MASK;
	    if (__half_as_ushort(out[idx]) & JAM_FLAG_MASK) {
		    if ((mi - jv) < mi) {
			    jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) - jv);
		    } else if ((mi + jv) > mi) {
			    jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) + jv);
		    } else {
			    jam[idx] = in[idx];
//			    jam[idx] = __ushort_as_half(0x3e00);
		    }
	    } else {
		    if ((mi + jv) > mi) {
			    jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) + jv);
		    } else if ((mi - jv) < mi) {
			    jam[idx] = __ushort_as_half(__half_as_ushort(in[idx]) - jv);
		    } else {
			    jam[idx] = in[idx];
//			    jam[idx] = __ushort_as_half(0x3f00);
		    }
	    }
	    //out[idx] = in[idx];
    } else {
	    jam[idx] = in[idx];
//	    jam[idx] = __ushort_as_half(0x3b00);
    }

  }

  sim[0] = 2;
}

#endif

}	// namespace


namespace functor {

template <typename T>
struct JamMeFunctor<GPUDevice, T> {
  void operator()(const OpKernelContext* ctx, const int size, const T* in, T* jam, T* out, float *sim) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();

    CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
    //JamMeRandCudaKernel_TODO<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, jam, out, sim);
    JamMeRandCudaKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, jam, out, sim);
  };
};

#define DEFINE_GPU_SPECS(T)	\
  template struct JamMeFunctor<GPUDevice, T>;


//TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
TF_CALL_half(DEFINE_GPU_SPECS);
TF_CALL_float(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

} // functor

} // tensorflow


#endif	// GOOGLE_CUDA


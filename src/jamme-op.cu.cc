
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


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

__global__ void JamMeRandCudaKernel(const int nthreads, const Eigen::half* in, Eigen::half* jam, Eigen::half* out, float *sim) {
	CUDA_1D_KERNEL_LOOP(idx, nthreads) {
		//unsigned short tmp;
		//unsigned short tmp1;
		unsigned short mask = (1<<JAM_BIT_NUM) -1;

		if (__half_as_ushort((__half_as_ushort((__half_as_ushort(in[idx])) << 1)) >> 11) == __half_as_ushort (0x1f)) { //NaN or Infinit
			jam[idx] = in[idx];
		} else if (in[idx] == (__ushort_as_half(0))) { // zero
			jam[idx] = in[idx];
		} else if ((__half_as_ushort(out[idx])%100) < JAM_BIT_RATIO) {
			if(((__half_as_ushort(in[idx])) & mask) == ((__half_as_ushort(out[idx])) & mask))
                                //tmp1 = (__half_as_ushort (out[idx])) >> 1;
				//tmp1 = ~(__half_as_ushort (out[idx]));
				jam[idx] = __ushort_as_half( (( __half_as_ushort(in[idx])) & (~mask)) | ( (~(__half_as_ushort(out[idx]))) & mask) );
			else
				jam[idx] = __ushort_as_half( (( __half_as_ushort(in[idx])) & (~mask)) | ( ( (__half_as_ushort(out[idx]))) & mask) );
				//tmp1 =__half_as_ushort (out[idx]);
			//tmp = ((__half_as_ushort(in[idx])) & (~mask)) | ((__half_as_ushort(out[idx])) & mask);
			//tmp = ((__half_as_ushort(in[idx])) & (~mask)) | ( tmp1 & mask); 
			//jam[idx] = __ushort_as_half(tmp);
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
}	// namespace


namespace functor {

template <typename T>
struct JamMeFunctor<GPUDevice, T> {
  void operator()(const OpKernelContext* ctx, const int size, const T* in, T* jam, T* out, float *sim) {
    const GPUDevice& d = ctx->eigen_device<GPUDevice>();

    CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
    JamMeRandCudaKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(size, in, jam, out, sim);
  };
};

#define DEFINE_GPU_SPECS(T)	\
  template struct JamMeFunctor<GPUDevice, T>;


TF_CALL_half(DEFINE_GPU_SPECS);
TF_CALL_float(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

} // functor

} // tensorflow


#endif	// GOOGLE_CUDA


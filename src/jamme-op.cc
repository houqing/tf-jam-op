
#define EIGEN_USE_THREADS

#include "jamme-op.h"

#include <stdlib.h>
#include <fstream>
#include <functional>
#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/core/status.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


#if 1
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/platform/stream_executor.h"

#endif	// GOOGLE_CUDA
#endif


namespace tensorflow {

REGISTER_OP("DumpMe")
.Input("in: T")
.Input("info: string")
.Output("out: T")
.Attr("T: {half, float, int32, int64}")
.SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("JamMe")
.Input("in: T")
//.Input("info: string")
.Output("jam: T")
.Output("out: T")
.Output("sim: float")
.Attr("T: {half, float}")
.SetShapeFn(shape_inference::UnchangedShape);


typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T>
class DumpMeOp : public AsyncOpKernel {
  public:
  explicit DumpMeOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
	  //std::cout << "==== Error: copy failed " << __LINE__ << std::endl;
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
	  const Tensor& in = ctx->input(0);
	  const Tensor& info = ctx->input(1);

	  ctx->set_output(0, in);
	  if (IsRefType(ctx->input_dtype(0))) {
		  ctx->forward_ref_input_to_ref_output(0, 0);
	  } else {
		  ctx->set_output(0, ctx->input(0));
	  }



	  string info_str = info.scalar<string>()();
	  if (info_str != "") {
		  auto device = static_cast<tensorflow::Device*>(ctx->device());

		  const char *my_dat;
		  size_t my_dat_size;
		  int is_copy_done = 0;
		  int *is_copy_done_p = &is_copy_done;
		  Allocator* cpu_allocator = tensorflow::cpu_allocator();
		  Tensor t_dat_cpu(cpu_allocator, in.dtype(), in.shape());

		  string shape_str = "d" + std::to_string(in.shape().dims()) + "";
		  for (size_t d = 0; d < in.shape().dims(); ++d) {
			  shape_str += "x" + std::to_string(in.shape().dim_sizes()[d]);
		  }


		  if ((device->name().find("GPU:") != string::npos
				  || device->name().find("SYCL:") != string::npos)
				  && !ctx->input_alloc_attr(0).on_host()) {
			  ctx->op_device_context()->CopyDeviceTensorToCPU(&in, "MyCopyFromGpuToHost", device, &t_dat_cpu,
					  [ctx, is_copy_done_p, done](const Status& s) {
					  ctx->SetStatus(s);
					  if (s.ok()) {
						  *is_copy_done_p = 1;
					  } else {
						  *is_copy_done_p = 2;
					  }
					  //done();
					  });

			  while (is_copy_done == 0) { }
			  my_dat = t_dat_cpu.tensor_data().data();
		  } else {
			  *is_copy_done_p = 1;
			  while (is_copy_done == 0) { }
			  t_dat_cpu.UnsafeCopyFromInternal(in, in.dtype(), in.shape());
			  my_dat = t_dat_cpu.tensor_data().data();

		  }
		  string filename;
		  if (is_copy_done == 1) {
			  filename = std::to_string(ctx->step_id()) + "--" + info_str + "-@" + std::to_string(file_id) + "-" + shape_str + ".bin";
		  } else {
			  std::cout << "==== Error: copy failed " << __LINE__ << std::endl;
			  filename = std::to_string(ctx->step_id()) + "--" + info_str + "-@" + std::to_string(file_id) + "-" + shape_str + "--DUMP_ERR.bin";
		  }
		 file_id++	;
		  std::ofstream fs_dat(filename, std::ofstream::binary);
		  my_dat_size = in.NumElements() * sizeof(T);
		  // TODO handle write error, long file name
		  fs_dat.write(my_dat, my_dat_size);
		  fs_dat.close();
	  }

	  done();
  }
  private:
  int file_id = 0;
};

template <typename Device, typename T>
class JamMeOp : public AsyncOpKernel {
  public:
  explicit JamMeOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
	  std::cout << "[jam_op] bit=" << JAM_BIT_NUM << " ratio=" << JAM_BIT_RATIO << std::endl;
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
	  const Tensor& in = ctx->input(0);
	  //const Tensor& info = ctx->input(1);

	  if (typeid(float) == typeid(T)) {
		  if (IsRefType(ctx->input_dtype(0))) {
			  ctx->forward_ref_input_to_ref_output(0, 0);
		  } else {
			  ctx->set_output(0, ctx->input(0));
		  }
		  Tensor* out = NULL;
		  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(1, in.shape(), &out), done);
		  Tensor* sim = NULL;
		  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(2, TensorShape({}), &sim), done);
		  done();
		  return;
	  }

	  Tensor* jam = NULL;
	  OP_REQUIRES_OK_ASYNC(ctx, ctx->forward_input_or_allocate_output({0}, 0, in.shape(), &jam), done);
	  //OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, in.shape(), &jam), done);
	  Tensor* out = NULL;
	  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(1, in.shape(), &out), done);
	  Tensor* sim = NULL;
	  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(2, TensorShape({}), &sim), done);

	  auto rand_data_float = perftools::gputools::DeviceMemory<float>::MakeFromByteSize(out->template flat<T>().data(), out->template flat<T>().size() * sizeof(T));
	  if (ctx->input_alloc_attr(0).on_host()) {
		  auto* stream = ctx->op_device_context()->stream();
		  OP_REQUIRES_ASYNC(ctx, stream, errors::Internal("No GPU stream avalible"), done);

		  bool launch_status = stream->ThenPopulateRandUniform(&rand_data_float).ok();
		  OP_REQUIRES_ASYNC(ctx, launch_status, errors::Internal("JamMe rand gen failed"), done);
	  } else {
	          //std::cout<<"Warning! the random data run failed!"<<std::endl;
	  }
          //int temp_i = 100000;
	  //while(temp_i > 0)
	  //{
		//  temp_i--;
	  //}
	  functor::JamMeFunctor<Device, T>()(
			  ctx,
			  static_cast<const int>(in.NumElements()),
			  in.flat<T>().data(),
			  jam->flat<T>().data(),
			  out->flat<T>().data(),
			  (float*)sim->flat<float>().data());

	  //std::cout << std::endl;

          //temp_i = 100000;
	  //while(temp_i > 0)
	  //{
	  //	  temp_i--;
	  //}

	  done();
  }

  private:
  int is_rand_initialized_ = false;
};


namespace functor {

namespace {

void JamMeRandCpuKernel(const int nthreads, const Eigen::half* in, Eigen::half* jam, Eigen::half* out, float *sim) {
	//std::cout <<" Warning: this is a Cpu JamMe function ! "<<std::endl;
	int idx;
	for (idx = 0; idx < nthreads; idx++) {
		//unsigned short tmp;
		unsigned short mask = (1<<JAM_BIT_NUM) -1;

		//out[idx] = (Eigen::half)((unsigned short)(rand() & 0xffff));
		//out[idx] = (Eigen::half)(rand() & 0xffff);
		out[idx] = (Eigen::half)((unsigned short)idx & 0xffff);
		if ((unsigned short)(((unsigned short)((unsigned short)(((unsigned short)(in[idx])) << 1))) >> 11) == 
				(unsigned short) 0x1f) { //NaN or Infinit
			jam[idx] = in[idx];
		} else if (in[idx] == ((Eigen::half)(0))) { // zero
			jam[idx] = in[idx];
		} else if ((((unsigned short)(out[idx]))%100) < JAM_BIT_RATIO) {
			if( ((unsigned short)(in[idx]) &  mask) == ( ((unsigned short)(out[idx])) & mask ) )
			        jam[idx] = (Eigen::half)((((unsigned short)(in[idx])) & (~mask)) | ( (~ ((unsigned short)(out[idx])) ) & mask));
			else
			        jam[idx] = (Eigen::half)((((unsigned short)(in[idx])) & (~mask)) | (((unsigned short)(out[idx])) & mask));
			//tmp = (((unsigned short)(in[idx])) & (~mask)) | (((unsigned short)(out[idx])) & mask);
			//jam[idx] = (Eigen::half)(tmp);
		} else {
			jam[idx] = in[idx];
		}
	}
}

void JamMeRandCpuKernel(const int nthreads, const float* in, float* jam, float* out, float *sim) {
	int idx;
	for (idx = 0; idx < nthreads; idx++) {
		jam[idx] = in[idx];
	}
}

}	// namespace

template <typename T>
struct JamMeFunctor<CPUDevice, T> {
  void operator()(const OpKernelContext* ctx, const int size, const T* in, T* jam, T* out, float *sim) {
    JamMeRandCpuKernel(size, in, jam, out, sim);
  };
};

#define DEFINE_CPU_SPECS(T)	\
  template struct JamMeFunctor<CPUDevice, T>;


TF_CALL_half(DEFINE_CPU_SPECS);
TF_CALL_float(DEFINE_CPU_SPECS);

#undef DEFINE_CPU_SPECS


#define REGISTER_KERNEL(T)	\
  REGISTER_KERNEL_BUILDER(Name("DumpMe")	\
		  .Device(DEVICE_CPU)	\
		  .TypeConstraint<T>("T"),	\
		  DumpMeOp<CPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)	\
  REGISTER_KERNEL_BUILDER(Name("JamMe")	\
		  .Device(DEVICE_CPU)	\
		  .TypeConstraint<T>("T"),	\
		  JamMeOp<CPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
#undef REGISTER_KERNEL


#if GOOGLE_CUDA

#define REGISTER_KERNEL(T)	\
  REGISTER_KERNEL_BUILDER(Name("DumpMe")	\
		  .Device(DEVICE_GPU)	\
		  .TypeConstraint<T>("T"),	\
		  DumpMeOp<GPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)	\
  REGISTER_KERNEL_BUILDER(Name("JamMe")	\
		  .Device(DEVICE_GPU)	\
		  .TypeConstraint<T>("T"),	\
		  JamMeOp<GPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#endif	// GOOGLE_CUDA

}	// namespace

}	// namespace tensorflow




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


#define JAM_ENABLE_DUMP_DATA	1

namespace tensorflow {

REGISTER_OP("DumpMe")
.Input("in: T")
.Input("info: string")
.Output("out: T")
.Attr("T: {half, float}")
.SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("JamMe")
.Input("in: T")
.Input("info: string")
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
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
	  const Tensor& in = ctx->input(0);
	  const Tensor& info = ctx->input(1);

	  ctx->set_output(0, in);

	  string info_str = info.scalar<string>()();
	  if (info_str != "") {
		  string filename = std::to_string(ctx->step_id()) + "#" + info_str + "__dmp";

		  OP_REQUIRES_ASYNC(ctx, !ctx->input_alloc_attr(0).on_host(),
				  errors::Internal("The input tensor to the _CopyFromGpuToHost kernel "
					  "must reside on the device."), done);

		  auto device = static_cast<tensorflow::Device*>(ctx->device());

		  AllocatorAttributes alloc_attrs_dat;
		  alloc_attrs_dat.set_gpu_compatible(true);
		  alloc_attrs_dat.set_on_host(true);
		  std::ofstream fs_dat(filename, std::ofstream::binary);

		  const char *my_dat;
		  size_t my_dat_size;
		  Tensor t_dat_cpu;
		  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(), in.shape(), &t_dat_cpu, alloc_attrs_dat), done);
		  ctx->op_device_context()->CopyDeviceTensorToCPU(&in, "CopyFromGpuToHost", device, &t_dat_cpu,
				  [ctx, done](const Status& s) {
				  ctx->SetStatus(s);
				  //done();
				  });
		  my_dat = t_dat_cpu.tensor_data().data();
		  my_dat_size = in.NumElements() * sizeof(T);
		  fs_dat.write(my_dat, my_dat_size);
		  fs_dat.close();
	  }

	  done();
  }
};

template <typename Device, typename T>
class JamMeOp : public AsyncOpKernel {
  public:
  explicit JamMeOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
	  const Tensor& in = ctx->input(0);
	  const Tensor& info = ctx->input(1);

#if 0
	  std::cout << "JamMeOp:"
		  << " info=" << filename_dat
		  << " step=" << ctx->step_id()
		  << " k.name=" << ctx->op_kernel().name()
		  << " k.type=" << ctx->op_kernel().type_string()
		  << std::endl;
#endif

	  Tensor* jam = NULL;
	  OP_REQUIRES_OK_ASYNC(ctx, ctx->forward_input_or_allocate_output({0}, 0, in.shape(), &jam), done);
	  Tensor* out = NULL;
	  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(1, in.shape(), &out), done);
	  Tensor* sim = NULL;
	  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(2, TensorShape({}), &sim), done);

	  auto rand_data_float = perftools::gputools::DeviceMemory<float>::MakeFromByteSize(out->template flat<T>().data(), out->template flat<T>().size() * sizeof(T));
	  auto* stream = ctx->op_device_context()->stream();
	  OP_REQUIRES_ASYNC(ctx, stream, errors::Internal("No GPU stream avalible"), done);

	  bool launch_status = stream->ThenPopulateRandUniform(&rand_data_float).ok();
	  OP_REQUIRES_ASYNC(ctx, launch_status, errors::Internal("JamMe rand gen failed"), done);

	  functor::JamMeFunctor<Device, T>()(
			  ctx,
			  static_cast<const int>(in.NumElements()),
			  in.flat<T>().data(),
			  jam->flat<T>().data(),
			  out->flat<T>().data(),
			  (float*)sim->flat<float>().data());

#ifdef JAM_ENABLE_DUMP_DATA
	  string info_str = info.scalar<string>()();
	  if (info_str != "") {
		  string filename_dat = std::to_string(ctx->step_id()) + "#" + info_str + "__dat";
		  string filename_jam = std::to_string(ctx->step_id()) + "#" + info_str + "__jam";

		  OP_REQUIRES_ASYNC(ctx, !ctx->input_alloc_attr(0).on_host(),
				  errors::Internal("The input tensor to the _CopyFromGpuToHost kernel "
					  "must reside on the device."), done);

		  auto device = static_cast<tensorflow::Device*>(ctx->device());

		  AllocatorAttributes alloc_attrs_dat;
		  alloc_attrs_dat.set_gpu_compatible(true);
		  alloc_attrs_dat.set_on_host(true);
		  std::ofstream fs_dat(filename_dat, std::ofstream::binary);
		  const char *my_dat;
		  size_t my_dat_size;
		  Tensor t_dat_cpu;
		  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(), in.shape(), &t_dat_cpu, alloc_attrs_dat), done);
		  ctx->op_device_context()->CopyDeviceTensorToCPU(&in, "CopyFromGpuToHost1", device, &t_dat_cpu,
				  [ctx, done](const Status& s) {
				  ctx->SetStatus(s);
				  //done();
				  });
		  my_dat = t_dat_cpu.tensor_data().data();
		  my_dat_size = in.NumElements() * sizeof(T);
		  fs_dat.write(my_dat, my_dat_size);
		  fs_dat.close();

		  if (typeid(T) != typeid(float)) {
			  AllocatorAttributes alloc_attrs_jam;
			  alloc_attrs_jam.set_gpu_compatible(true);
			  alloc_attrs_jam.set_on_host(true);
			  std::ofstream fs_jam(filename_jam, std::ofstream::binary);
			  const char *my_jam;
			  size_t my_jam_size;
			  Tensor t_jam_cpu;
			  OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(), in.shape(), &t_jam_cpu, alloc_attrs_jam), done);
			  ctx->op_device_context()->CopyDeviceTensorToCPU(jam, "CopyFromGpuToHost2", device, &t_jam_cpu,
					  [ctx, done](const Status& s) {
					  ctx->SetStatus(s);
					  //done();
					  });
			  my_jam = t_jam_cpu.tensor_data().data();
			  my_jam_size = in.NumElements() * sizeof(T);
			  fs_jam.write(my_jam, my_jam_size);
			  fs_jam.close();
		  }
	  }
#endif

	  done();
  }

  private:
  int is_rand_initialized_ = false;
};


namespace functor {


#if GOOGLE_CUDA

#define REGISTER_KERNEL(T)	\
  REGISTER_KERNEL_BUILDER(Name("JamMe")	\
		  .Device(DEVICE_GPU)	\
		  .TypeConstraint<T>("T"),	\
		  JamMeOp<GPUDevice, T>);	\
  REGISTER_KERNEL_BUILDER(Name("DumpMe")	\
		  .Device(DEVICE_GPU)	\
		  .TypeConstraint<T>("T"),	\
		  DumpMeOp<GPUDevice, T>);

//TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif	// GOOGLE_CUDA

}	// namespace

}	// namespace tensorflow



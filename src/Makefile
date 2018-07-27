
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')


all: a b

test: t

clean:
	rm -f jamme-op.cu.o jamme-op.so

t:
	python ../example/jamme-op-example.py

a:
	nvcc -std=c++11 -c -o jamme-op.cu.o jamme-op.cu.cc ${TF_CFLAGS[@]} $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
		-I/usr/local \
		--expt-relaxed-constexpr \
		-D__CUDA_NO_HALF_OPERATORS__

b:
	g++ -std=c++11 -shared -o jamme-op.so jamme-op.cc jamme-op.cu.o ${TF_CFLAGS[@]} $(TF_CFLAGS) -fPIC -lcudart ${TF_LFLAGS[@]} $(TF_LFLAGS) \
		-L/usr/local/cuda/lib64 -D GOOGLE_CUDA=1



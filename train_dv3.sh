#!/bin/bash
# __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python scripts/train_dreamerv3.py "$@"
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_bf16=false"

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia JAX_PLATFORMS=cuda python ./dreamerv3/dreamerv3/main.py --configs parallel_parking --logdir ./logs/parallel_parking "$@"
# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import sys
import torch
from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if torch.version.hip is not None:
    # ROCm: BuildExtension hipifies kernels.cu and injects the target arch
    # via --offload-arch (PYTORCH_ROCM_ARCH). The nvcc -gencode flags,
    # --ptxas-options and --use_fast_math are CUDA-only; hipcc enables fast
    # math / FMA contraction by default, so -O3 suffices.
    nvcc_args = ['-O3']
else:
    # compile for all possible CUDA architectures
    all_cuda_archs = cuda.get_gencode_flags().replace('compute=','arch=').split()
    # alternatively, you can list cuda archs that you want, eg:
    # all_cuda_archs = [
        # '-gencode', 'arch=compute_70,code=sm_70',
        # '-gencode', 'arch=compute_75,code=sm_75',
        # '-gencode', 'arch=compute_80,code=sm_80',
        # '-gencode', 'arch=compute_86,code=sm_86'
    # ]
    nvcc_args = ['-O3','--ptxas-options=-v',"--use_fast_math"]+all_cuda_archs

extra_link_args = []
if sys.platform == "win32":
    # c10.dll does not export c10::ValueError(SourceLocation, string) --
    # MSVC does not re-export inherited constructors for dllexport classes.
    # TORCH_CHECK in curope.cpp generates a dllimport reference that causes
    # LNK2001.  Alias to Error(SourceLocation, string) which IS exported.
    _val_imp = (
        "__imp_??0ValueError@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    _err_imp = (
        "__imp_??0Error@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    extra_link_args.append(f"/ALTERNATENAME:{_val_imp}={_err_imp}")

setup(
    name = 'curope',
    ext_modules = [
        CUDAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.cu",
                ],
                extra_compile_args = dict(
                    nvcc=nvcc_args,
                    cxx=['-O3']),
                extra_link_args=extra_link_args,
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })

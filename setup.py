from pathlib import Path
from setuptools import setup
import sys

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
has_cuda = torch.cuda.is_available()

include_dirs = [
    os.path.join(ROOT, "mast3r_slam/backend/include"),
    os.path.join(ROOT, "thirdparty/eigen"),
]

sources = [
    "mast3r_slam/backend/src/gn.cpp",
]
extra_compile_args = {
    "cores": ["j8"],
    "cxx": ["-O3"],
}

extra_link_args = []
if sys.platform == "win32":
    # c10.dll does not export c10::ValueError(SourceLocation, string) -- the
    # constructor is inherited via "using Error::Error" and MSVC does not
    # re-export inherited constructors for __declspec(dllexport) classes.
    # Headers pulled in by <torch/extension.h> trigger TORCH_CHECK which
    # generates a __declspec(dllimport) reference to that constructor, causing
    # LNK2001.  Redirect the missing thunk to Error(SourceLocation, string)
    # which IS exported from c10.dll.  ValueError IS-A Error with no extra
    # data members; the constructors are semantically identical.
    _val_imp = (
        "__imp_??0ValueError@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    _err_imp = (
        "__imp_??0Error@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    extra_link_args.append(f"/ALTERNATENAME:{_val_imp}={_err_imp}")

if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension

    sources.append("mast3r_slam/backend/src/gn_kernels.cu")
    sources.append("mast3r_slam/backend/src/matching_kernels.cu")
    if torch.version.hip is not None:
        # ROCm: BuildExtension hipifies the .cu sources and injects the
        # target arch via --offload-arch (PYTORCH_ROCM_ARCH); the nvcc
        # -gencode flags are CUDA-only and rejected by hipcc.
        extra_compile_args["nvcc"] = ["-O3"]
    else:
        extra_compile_args["nvcc"] = [
            "-O3",
            "-gencode=arch=compute_60,code=sm_60",
            "-gencode=arch=compute_61,code=sm_61",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ]
    ext_modules = [
        CUDAExtension(
            "mast3r_slam_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
else:
    print("CUDA not found, cannot compile backend!")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

from pathlib import Path
from setuptools import setup

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

if has_cuda:
    from torch.utils.cpp_extension import CUDAExtension
    from torch import cuda as _cuda

    # Determine local GPU compute capability and build only for it
    cc_major, cc_minor = _cuda.get_device_capability(0)
    cc = f"{cc_major}{cc_minor}"

    sources.append("mast3r_slam/backend/src/gn_kernels.cu")
    sources.append("mast3r_slam/backend/src/matching_kernels.cu")

    arch_flags = [
        f"-gencode=arch=compute_{cc},code=sm_{cc}",
        f"-gencode=arch=compute_{cc},code=compute_{cc}",
    ]

    extra_compile_args["nvcc"] = [
        "-O3",
        "--use_fast_math",
        "--ptxas-options=-v",
        *arch_flags,
    ]

    ext_modules = [
        CUDAExtension(
            "mast3r_slam_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    print("CUDA not found, cannot compile backend!")

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)

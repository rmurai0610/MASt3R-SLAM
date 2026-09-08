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

    sources.append("mast3r_slam/backend/src/gn_kernels.cu")
    sources.append("mast3r_slam/backend/src/matching_kernels.cu")
    extra_compile_args["nvcc"] = [
        "-O3",
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
        # RTX 5090 / Blackwell: deliberately NOT compiling native sm_120 SASS
        # here. gauss_newton_rays's ray_align_kernel produces an intermittent
        # illegal-memory-access with provably in-bounds inputs when compiled
        # native for sm_120 (root-caused via compute-sanitizer + Python-side
        # bounds prints -- see SETUP_TROUBLESHOOTING.md item 18) -- suspected
        # nvcc/ptxas codegen bug specific to this very new architecture.
        # Embedding compute_90 PTX instead makes the CUDA driver JIT-compile
        # it for sm_120 at first kernel launch (a different code path/JIT
        # compiler than nvcc's ahead-of-time sm_120 codegen), as an
        # experiment to route around the suspected bug.
        "-gencode=arch=compute_90,code=compute_90",
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

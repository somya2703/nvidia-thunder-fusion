import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="fused_kernels",
    ext_modules=[
        CUDAExtension(
            name="fused_kernels",
            sources=["kernels/conv_bn_relu_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

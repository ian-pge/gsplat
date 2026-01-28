import glob
import os
import os.path as osp
import pathlib
import platform
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

# PATCH: Set CUDA_HOME
import sys
import os

print(f"DEBUG: sys.executable = {sys.executable}")
print(f"DEBUG: CUDA_HOME initial = {os.environ.get('CUDA_HOME')}")

candidate_home = os.path.dirname(os.path.dirname(sys.executable))
nvcc_candidate = os.path.join(candidate_home, "bin", "nvcc")

if os.path.exists(nvcc_candidate):
    print(f"FORCE SETTING CUDA_HOME to {candidate_home}")
    os.environ["CUDA_HOME"] = candidate_home
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
         os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;9.0+PTX"
         print("FORCE SETTING TORCH_CUDA_ARCH_LIST")
    
    # Force import and monkey patch
    try:
        import torch.utils.cpp_extension
        print("Force imported torch.utils.cpp_extension")
        torch.utils.cpp_extension.CUDA_HOME = candidate_home
        
        def mock_find_cuda_home():
            print(f"Mock _find_cuda_home called, returning {candidate_home}")
            return candidate_home
            
        torch.utils.cpp_extension._find_cuda_home = mock_find_cuda_home
        print("Monkey patched torch.utils.cpp_extension._find_cuda_home")
    except Exception as e:
        print(f"Failed to patch torch: {e}")
else:
    print("WARNING: nvcc not found in conda env!")

sys.stdout.flush()






URL = "https://github.com/nerfstudio-project/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("gsplat", "cuda")
    sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
        osp.join(extensions_dir, "csrc", "*.cpp")
    )
    sources += [osp.join(extensions_dir, "ext.cpp")]

    undef_macros = []
    define_macros = []

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math", "-std=c++17"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]
    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["--expt-relaxed-constexpr"]

    # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
    nvcc_flags += ["-diag-suppress", "20012,186"]
    extra_compile_args["nvcc"] = nvcc_flags
    if sys.platform == "win32":
        extra_compile_args["nvcc"] += [
            "-DWIN32_LEAN_AND_MEAN",
            "-allow-unsupported-compiler",
        ]

    current_dir = pathlib.Path(__file__).parent.resolve()
    glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
    include_dirs = [glm_path, osp.join(current_dir, "gsplat", "cuda", "include")]

    extension = CUDAExtension(
        "gsplat.csrc",
        sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    return [extension]


setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml>=6.0.1",
            "build",
            "twine",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")

# NATTEN Windows wheel

This directory contains the NATTEN CUDA wheel built for the manager's pinned
PyTorch stack, together with its source patch and validation results. The manager
automatically installs it as an optional accelerator on matching Windows systems.

## Compatibility

| Component | Build target |
| --- | --- |
| NATTEN | 0.21.7 |
| Platform | Windows x64 (`win_amd64`) |
| Python | CPython 3.12 (`cp312`) |
| PyTorch | 2.13.0+cu130 |
| CUDA Toolkit | 13.0.0 (`nvcc` 13.0.48) |
| GPU architecture | SM 12.0 only; no PTX fallback |
| Validation GPU | NVIDIA GeForce RTX 5080 |
| NVIDIA driver used for validation | 591.86 |
| C++ compiler | MSVC 19.44.35229 (VS 2022 toolset) |
| Windows SDK | 10.0.26100.0 |
| CMake / Ninja | 4.1.0 / 1.13.2 |

Do not install this binary into a different Python or PyTorch/CUDA stack, or use
it on GPUs with a different compute capability. The wheel's Python/platform tags
do not enforce its PyTorch or GPU compatibility.

## Manager integration

The manager selects this exact local wheel when it uses Windows x64, Python
3.12, PyTorch 2.13.0/CUDA 13.0, and every detected GPU reports SM 12.0. The x64
fallback on Windows ARM uses the same selection rules. Native ARM builds use
their separate Spark wheel source.

Other GPU architectures, unknown compute capabilities, CUDA 12.6 environments,
missing wheel files, and future Python/PyTorch/NATTEN pin changes skip this wheel.
The manager does not fall back to a wheel built for an older PyTorch ABI.
After installation it checks `natten.HAS_LIBNATTEN` and removes the optional
package if its compiled extension cannot load.

Run the normal manager install/update flow to install it. The selected wheel
appears in `python -m manager detect --json` under `spec.optional_packages`.

## Manual install

From the repository root, in the matching environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps .\wheels\natten\natten-0.21.7+torch2130cu130-cp312-cp312-win_amd64.whl
.\.venv\Scripts\python.exe -c "import natten; print(natten.HAS_LIBNATTEN)"
.\.venv\Scripts\python.exe .\wheels\natten\validate.py
```

`HAS_LIBNATTEN` must be `True`. `SHA256SUMS` records the artifact checksum.

## Rebuild

Source: https://github.com/SHI-Labs/NATTEN, tag `v0.21.7`, commit
`39375ea1d2f3b1bb284315fa5f79f6ecbebb6382`.
CUTLASS submodule: `6b3e607b852f1543dc21323155a2ad71473c8642`.

Use an x64 VS 2022 developer shell with the compatible CUDA Toolkit on PATH,
and a Python 3.12 environment containing `torch==2.13.0+cu130`.
Install `cmake==4.1.0`, `ninja==1.13.2`, `setuptools==84.0.0`,
`wheel==0.48.0`, and `build==1.6.1` in that build environment.

```powershell
git clone --recursive --branch v0.21.7 https://github.com/SHI-Labs/NATTEN.git
cd NATTEN
git apply /path/to/AITK-Studio/wheels/natten/windows-build.patch
$env:NATTEN_CUDA_ARCH = '12.0'
$env:NATTEN_N_WORKERS = '16'
$env:NATTEN_IS_BUILDING_DIST = '1'
python setup.py bdist_wheel
```

The patch replaces Linux-only compiler options on Windows, uses the base Python
installation's import libraries, handles extension output paths correctly for
CMake/Ninja, links CUDA's runtime through its CMake target, and replaces one
`not` token with equivalent `!` syntax accepted by the Windows CUDA compiler.
For PyTorch 2.13 and newer it selects C++20 for both host and CUDA compilation,
as required by the newer headers. Kernel algorithms are unchanged. VS 2026's
compiler failed CUDA 13.0's compiler check on this machine; the build uses VS 2022.

## Validation

Validation on 2026-09-20: all 30 forward/backward comparisons, four selected
upstream tests, and ten manager regression tests passed. The manager's installer
also selected and installed the wheel in the isolated PyTorch 2.13 environment.
See `validation.txt` for the recorded results.
`validate.py` checks compiled
1D/2D/3D neighborhood attention and full attention, including forward outputs
and query/key/value gradients, using FP16, BF16, and FP32. It includes causal
and dilated neighborhood cases. These checks are not an exhaustive upstream
test suite or validation on other GPUs. The upstream tests cover compute-delta
and the quick randomized 1D/2D/3D FNA comparisons against reference kernels:

```powershell
python -m pytest -q tests/test_compute_delta.py tests/test_fna.py -k "test_against_pt_reference or quick"
```

## Licenses

`LICENSE-NATTEN.txt` and `NOTICE-NATTEN.txt` are from the pinned NATTEN source;
`LICENSE-CUTLASS.txt` is from its pinned CUTLASS submodule.
`LICENSE-xFormers.txt` supplies the license for the xFormers-derived kernels
identified by NATTEN's notice. `LICENSE-CUDA.txt` is the CUDA Toolkit license
distributed with the build components. Keep these notices with redistributed binaries.

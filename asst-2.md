# E0358 Aug-Dec 2026: Assignment 2

**Deadline: Friday Nov 27, 5 pm.**

## Description

Building upon what we did for Asst-1, this assignment requires the development
of a compilation path that uses NVIDIA CUDA Tile IR for the lower part of the
compilation.

- https://developer.nvidia.com/cuda/tile
- https://github.com/NVIDIA/cuda-tile

The scope of optimization is restricted primarily to the specification and
optimization asked for further below. The first goal is to be able to lower
matmul in a performant way using CUDA Tile IR. In particular, you are expected
to build a pass pipeline that lowers an affine matmul nest (lowered from PyTorch
matmul's) to CUDA Tile IR.

The second step will be to add the ability to perform "sibling fusion", which
exploits input tensor reuse, on multiple matmuls that share an operand.

In particular, consider the three matmuls that compute the Q, K, V projection
matrices for an attention layer, where `x` is `N x d_model` and `W_q`, `W_k`,
`W_v` are `d_model x d_q`, `d_model x d_k`, `d_model x d_v` respectively:

```
Q = x dot W_q
K = x dot W_k
V = x dot W_v
```

These share the same operand `x` and the goal is to be able to fuse the three to
reuse the operand for all matmuls and obtain a performance improvement over not
fusing.

## Test harness

See `qkv-projection.py`, which provides the harness to test this; `common.py`
has the command-line parsing, backend lookup, validation, and timing code it
uses. The `e0358` backend designed should be able to accomplish the fusion while
utilizing the CUDA Tile IR backend to also achieve high performance for the
matmuls.

The harness runs the projections at three problem sizes drawn from common
attention layer configurations, with `d_q = d_k = d_v = d`:

| size        |    N | d_model |    d | configuration                      |
|-------------|-----:|--------:|-----:|------------------------------------|
| bert-base   |  512 |     768 |   64 | BERT-base, one of its 12 heads     |
| gpt2-medium | 1024 |    1024 | 1024 | GPT-2 medium, all 16 heads at once |
| llama-7b    | 4096 |    4096 |  128 | LLaMA-7B, one of its 32 heads      |

For every size, the harness

1. validates the outputs of the function compiled with the selected backend
   against a reference -- eager execution by default, or Inductor with
   `--reference inductor` -- and
2. benchmarks the baselines -- eager execution, Inductor, and TensorRT (through
   Torch-TensorRT) -- and the selected backend, which for `e0358` is run both
   without and with sibling fusion, running warmup iterations followed by timed
   steady-state iterations and reporting their median (or mean, with
   `--stat mean`) along with the achieved TFLOP/s.

Run it with:

```sh
python qkv-projection.py --backend e0358      # All measurements, all sizes.
python qkv-projection.py --backend e0358 --fusion on --baselines  # e0358 only.
python qkv-projection.py                      # Baselines only.
python qkv-projection.py --help               # All options.
```

With `--backend e0358`, the harness prints a table with one row for each of
eager, Inductor, TensorRT, e0358 without fusion, and e0358 with fusion, and one
column per problem size: exactly the measurements asked for under "What to
submit" below.

Use `--sizes` to run a subset of the sizes, `--iters 0` to only validate,
`--warmup` and `--iters` to change the iteration counts, `--dtype` to run in
float16 or bfloat16 instead of the default float32, `--baselines` to choose the
baselines to run (none, if given no names), and `--fusion on` or `--fusion off`
to run the e0358 backend under a single fusion setting.

The harness sets the environment variable `E0358_SIBLING_FUSION` to `1` or `0`
immediately before invoking your backend to compile the graph, and compiles the
function once under each setting. Your backend should therefore read the
variable every time it is invoked to compile a graph (not just once, when it is
imported) and perform sibling fusion only when it is `1`; this is what makes the
performance with and without fusion comparable on the same code. As in Asst-1,
the harness imports a module named `e0358`, which should register the backend
with `torch._dynamo` (e.g., via `torch._dynamo.register_backend(name="e0358")`).
Fusion can be verified through nsys profiling (a single kernel for the three
matmuls instead of three) or by inspecting the IR at the exit of your optimizer.

The `tensorrt` baseline uses Torch-TensorRT (`pip install torch-tensorrt`,
matching your PyTorch version), which registers the `tensorrt` backend with
`torch._dynamo` when imported. The harness compiles it with `min_block_size=1`
and `require_full_compilation=True`, so that the three matmuls actually run in
TensorRT, and with TF32 disabled for float32 to match PyTorch's default float32
matmul precision.

## Versions and formatting

CUDA Tile IR version 13.4.3 (tag `v13.4.3` of the cuda-tile repository above,
the latest of its `v13.4.x` tags) is to be used. cuda-tile builds against a
specific LLVM commit, specified in its `cmake/IncludeLLVM.cmake`; for v13.4.3
this is

```
d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde    (llvm-project, 2026-11-03)
```

So that no patches to cuda-tile are needed, this assignment uses LLVM/MLIR at
exactly that commit instead of the `llvmorg-23.1.3` tag used in Asst-1. Check
out the commit above in llvm-project and rebase your Asst-1 MLIR changes onto
it; 23.1.3 branched off main in January 2026, so expect to adapt your changes
to API changes made since then.

Switching to a more recent LLVM also requires switching stablehlo, which is
tied to a specific LLVM commit (`build_tools/llvm_version.txt`) and no longer
builds against the LLVM commit above at the commit used in Asst-1. Use stablehlo
at commit

```
b26a2e2a6e91b9144f8dc1aafbee74d01016d558    (stablehlo, 2026-11-04)
```

which integrates LLVM at `c744694c6ae1`, the last stablehlo integration before
the LLVM commit above; none of the MLIR changes between the two affect
stablehlo, so it builds against the LLVM commit above unchanged.

Build LLVM/MLIR as in Asst-1 (a Release build with assertions, LLD, MLIR, and
the X86 and NVPTX targets), then stablehlo and cuda-tile against that build:

```sh
# LLVM/MLIR
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde
cmake -G Ninja -S llvm -B build -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_LLD=ON -DLLVM_INSTALL_UTILS=ON
ninja -C build

# stablehlo
git clone https://github.com/openxla/stablehlo.git
cd stablehlo && git checkout b26a2e2a6e91b9144f8dc1aafbee74d01016d558
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=<llvm-project>/build/lib/cmake/mlir
ninja -C build

# cuda-tile
git clone https://github.com/NVIDIA/cuda-tile.git
cd cuda-tile && git checkout v13.4.3
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TILE_USE_LLVM_INSTALL_DIR=<llvm-project>/build
ninja -C build
```

The cuda-tile Python bindings (`-DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON`) are
optional; they require MLIR to have been built with
`-DMLIR_ENABLE_BINDINGS_PYTHON=ON`. With the versions above, no patches to
cuda-tile or stablehlo should be needed.

As in Asst-1, submit all your MLIR additions as a single git commit patch, now
on top of the LLVM commit above (`git format-patch HEAD~`), formatted with
clang-format (`git clang-format HEAD~`). All Python code submitted should be
formatted with black and wrapped at 80 cols.

## What to submit

Submit a tarball (tar.gz file for example) that includes:

- your compiler: the `e0358` Python module and everything it needs,
- your LLVM/MLIR changes as a single `git format-patch` on top of LLVM commit
  `d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde`, and
- a README.md.

The README.md should state the versions of LLVM, stablehlo, and cuda-tile used
(they should be the ones above; if you nevertheless had to patch cuda-tile or
stablehlo, include and explain the patches), list the environment variables to
set and the steps to build and run your compiler so that `qkv-projection.py`
can be run with the `e0358` backend at all problem sizes, and explain the
design of your lowering to CUDA Tile IR and of the sibling fusion, including
how the fusion decision is made.

The README.md should report all measured performances, with eager, inductor,
TensorRT, and your e0358 backend on the test cases for all problem sizes and
comment on the performance obtained. Include performance with the e0358 backend
with and without fusion. The GPU used should be an NVIDIA RTX 4090 available at
CSA and for which you already have access/can request access.

The comparison should include at least the following measurements for all the
problem sizes (these are the rows that `qkv-projection.py --backend e0358`
reports):

- PyTorch eager
- PyTorch inductor
- PyTorch with TensorRT backend
- PyTorch e0358 (no fusion)
- PyTorch e0358 (with fusion)

Submit to udayb@iisc.ac.in with subject: "E0358 Asst-2 submission". Do not
change the subject.

## Use of AI coding agents

The use of AI coding agents is only allowed as explained in class for some of
the lower level implementation components. They are not to be used for the
high-level design and overall approach at the top level. The submission should
have a section describing which parts were AI-assisted, including all the
prompts used as is.

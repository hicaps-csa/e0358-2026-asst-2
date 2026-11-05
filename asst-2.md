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
2. benchmarks eager execution, Inductor, and the selected backend, running
   warmup iterations followed by timed steady-state iterations and reporting
   their median (or mean, with `--stat mean`) along with the achieved TFLOP/s.

Run it with:

```sh
python qkv-projection.py                             # Inductor (reference).
python qkv-projection.py --backend e0358             # Your backend.
python qkv-projection.py --backend e0358 --fusion off
python qkv-projection.py --help                      # All options.
```

Use `--sizes` to run a subset of the sizes, `--iters 0` to only validate,
`--warmup` and `--iters` to change the iteration counts, and `--dtype` to run in
float16 or bfloat16 instead of the default float32.

The harness sets the environment variable `E0358_SIBLING_FUSION` to `1`
(`--fusion on`, the default) or `0` (`--fusion off`) before compiling. Your
backend should read it and perform sibling fusion only when it is `1`, so that
the performance with and without fusion can be compared on the same code. As in
Asst-1, the harness imports a module named `e0358`, which should register the
backend with `torch._dynamo` (e.g., via
`torch._dynamo.register_backend(name="e0358")`). Fusion can be verified through
nsys profiling (a single kernel for the three matmuls instead of three) or by
inspecting the IR at the exit of your optimizer.

## Versions and formatting

Continue to use LLVM 23.1.3 (`llvmorg-23.1.3`) and the stablehlo commit
specified in Asst-1 as the base. As in Asst-1, submit all your MLIR additions as
a single git commit patch on top of `llvmorg-23.1.3` (`git format-patch HEAD~`),
formatted with clang-format (`git clang-format HEAD~`).

CUDA Tile IR 13.4 (the `v13.4.x` tags of the cuda-tile repository above) is
recommended. Note that cuda-tile builds against a specific LLVM commit (see its
`cmake/IncludeLLVM.cmake`), which need not match `llvmorg-23.1.3`. A different
version of CUDA Tile can be used to align with the LLVM/MLIR version that has
already been used; the README.md should describe the Tile IR version that is
needed, and if necessary, the submission can include additional patches that
were necessary to apply on the specified CUDA Tile version.

All Python code submitted should be formatted with black and wrapped at 80 cols.

## What to submit

Submit a tarball (tar.gz file for example) that includes:

- your compiler: the `e0358` Python module and everything it needs,
- your LLVM/MLIR changes as a single `git format-patch` on top of
  `llvmorg-23.1.3`,
- any patches you applied to CUDA Tile IR, and
- a README.md.

The README.md should state the CUDA Tile IR version used (and describe the
patches applied to it, if any), list the environment variables to set and the
steps to build and run your compiler so that `qkv-projection.py` can be run with
the `e0358` backend at all problem sizes, and explain the design of your
lowering to CUDA Tile IR and of the sibling fusion, including how the fusion
decision is made.

The README.md should report all measured performances, with eager, inductor, and
your e0358 backend on the test cases for all problem sizes and comment on the
performance obtained. Include performance with the e0358 backend with and
without fusion. The GPU used should be an NVIDIA RTX 4090 available at CSA and
for which you already have access/can request access.

The comparison should include at least the following measurements for the problem sizes:

PyTorch eager
PyTorch inductor
PyTorch with TensorRT backend
PyTorch e0358 (no fusion)
PyTorch e0358 (with fusion)

Submit to udayb@iisc.ac.in with subject: "E0358 Asst-2 submission". Do not
change the subject.

## Use of AI coding agents

The use of AI coding agents is only allowed as explained in class for some of
the lower level implementation components. They are not to be used for the
high-level design and overall approach at the top level. The submission should
have a section describing which parts were AI-assisted, including all the
prompts used as is.

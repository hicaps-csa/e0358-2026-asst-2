"""
Utilities shared by the E0358 Assignment 2 test scripts.

A test script defines a workload (a function to compile, a routine that
builds its inputs, and the problem sizes to run it at) and defers the rest
to this module: command-line parsing, resolving the torch.compile backend,
validating the compiled outputs against a reference (eager or Inductor),
and benchmarking with warmup iterations followed by timed steady-state
iterations.
"""

import argparse
import os
import statistics
import time

import torch

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Tolerances for validating compiled outputs against the reference. These are
# looser than torch.testing's defaults to allow for a different accumulation
# order in matmuls with large reduction dimensions.
TOLERANCES = {
    torch.float32: dict(rtol=1e-4, atol=1e-3),
    torch.float16: dict(rtol=1e-2, atol=2e-2),
    torch.bfloat16: dict(rtol=2e-2, atol=5e-2),
}

# The harness sets this environment variable from --fusion before compiling;
# the e0358 backend reads it to decide whether to perform sibling fusion.
FUSION_ENV_VAR = "E0358_SIBLING_FUSION"


def get_backend(name):
    """Returns the torch.compile backend to use for 'name'."""
    if name == "e0358":
        # The E0358 backend is expected to register itself with Dynamo (e.g.,
        # via torch._dynamo.register_backend(name="e0358")) when imported.
        try:
            import e0358  # noqa: F401
        except ImportError:
            pass
        if "e0358" not in torch._dynamo.list_backends():
            raise SystemExit(
                "Backend 'e0358' is not registered with torch._dynamo; make "
                "sure the 'e0358' module is importable and registers it."
            )
    return name


def parse_args(description, sizes):
    """Parses the command-line flags common to all test scripts. 'sizes' maps
    problem-size names to the sizes themselves."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backend",
        default="inductor",
        help="torch.compile backend to test: inductor (default) or e0358",
    )
    parser.add_argument(
        "--reference",
        default="eager",
        choices=["eager", "inductor"],
        help="what to validate the compiled outputs against (default: eager)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=sorted(DTYPES),
        help="element type of all tensors (default: float32)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(sizes),
        choices=list(sizes),
        metavar="SIZE",
        help=f"problem sizes to run (default: all): {', '.join(sizes)}",
    )
    parser.add_argument(
        "--fusion",
        default="on",
        choices=["on", "off"],
        help=f"sets {FUSION_ENV_VAR}=1/0 for the e0358 backend (default: on)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="untimed warmup iterations before timing (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="timed steady-state iterations (default: 50; 0 to only validate)",
    )
    parser.add_argument(
        "--stat",
        default="median",
        choices=["median", "mean"],
        help="statistic of the timed iterations to report (default: median)",
    )
    parser.add_argument(
        "--no-inductor",
        action="store_true",
        help="do not benchmark Inductor (unless it is the backend/reference)",
    )
    return parser.parse_args()


def check(actual, expected, dtype):
    """Compares the output tuples 'actual' and 'expected'. Returns whether
    they match within tolerance, the largest absolute error, and the
    message of the first mismatch (or None)."""
    assert len(actual) == len(expected)
    ok, max_err, message = True, 0.0, None
    for i, (a, e) in enumerate(zip(actual, expected)):
        max_err = max(max_err, (a.float() - e.float()).abs().max().item())
        try:
            torch.testing.assert_close(a, e, **TOLERANCES[dtype])
        except AssertionError as err:
            if ok:
                message = f"Output {i} mismatch: {err}"
            ok = False
    return ok, max_err, message


def timeit(fn, args, device, warmup, iters):
    """Runs 'fn(*args)' for 'warmup' untimed iterations followed by 'iters'
    timed ones, and returns the per-iteration times in ms."""
    for _ in range(warmup):
        fn(*args)
    times = []
    if device == "cuda":
        # Record a pair of events around every iteration and synchronize
        # once at the end, so that timing does not perturb the steady state.
        torch.cuda.synchronize()
        events = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        times = [start.elapsed_time(end) for start, end in events]
    else:
        for _ in range(iters):
            start = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - start) * 1e3)
    return times


def summarize(times, stat):
    """Returns the requested statistic ('median' or 'mean') of 'times'."""
    if stat == "median":
        return statistics.median(times)
    return statistics.fmean(times)


def print_table(rows, variants, args, flops):
    """Prints the benchmark results. 'rows' is a list of (name, size, times)
    with 'times' mapping each variant to its time in ms."""
    print()
    print(
        f"{args.stat} of {args.iters} iterations after {args.warmup} warmup "
        f"iterations; time in ms"
        + (" (TFLOP/s in parentheses)" if flops else "")
        + "."
    )
    width = max(len(v) for v in variants) + 10
    header = f"{'size':<12} {'N':>6} {'d_model':>8} {'d':>6}"
    header += "".join(f"{v:>{width}}" for v in variants)
    print(header)
    print("-" * len(header))
    for name, size, times in rows:
        line = f"{name:<12} {size.N:>6} {size.d_model:>8} {size.d:>6}"
        for v in variants:
            cell = f"{times[v]:.3f}"
            if flops:
                cell += f" ({flops(size) / times[v] / 1e9:.1f})"
            line += f"{cell:>{width}}"
        print(line)


def run(fn, make_inputs, sizes, args, flops=None):
    """For each selected problem size: builds inputs, validates 'fn' compiled
    with args.backend against the reference, and benchmarks eager execution,
    Inductor, and the backend.

    'make_inputs(size, device, dtype)' returns the tuple of arguments to call
    'fn' with, and 'fn' returns a tuple of tensors. 'flops(size)', if given,
    returns the number of floating-point operations 'fn' performs, used to
    report the achieved TFLOP/s.
    """
    dtype = DTYPES[args.dtype]
    os.environ[FUSION_ENV_VAR] = "1" if args.fusion == "on" else "0"
    backend = get_backend(args.backend)
    print(
        f"backend={args.backend} reference={args.reference} "
        f"fusion={args.fusion} device={args.device} dtype={args.dtype}"
    )

    # The variants to benchmark, in column order.
    variants = ["eager"]
    if (
        not args.no_inductor
        or args.backend == "inductor"
        or args.reference == "inductor"
    ):
        variants.append("inductor")
    if args.backend != "inductor":
        variants.append(args.backend)

    rows, failed = [], []
    for name in args.sizes:
        size = sizes[name]
        inputs = make_inputs(size, args.device, dtype)

        # Compile afresh for every size with static shapes, so that each
        # size is compiled on its own and not through a graph specialized
        # (or made dynamic) for a previous one.
        torch._dynamo.reset()
        fns = {"eager": fn}
        for v in variants[1:]:
            fns[v] = torch.compile(fn, backend=get_backend(v), dynamic=False)
        compiled = fns[args.backend]

        expected = fns[args.reference](*inputs)
        actual = compiled(*inputs)
        ok, max_err, message = check(actual, expected, dtype)
        print(
            f"[{args.backend}/{args.device}] {name}: "
            f"{'PASSED' if ok else 'FAILED'} vs {args.reference} "
            f"({len(expected)} outputs of shape {tuple(expected[0].shape)}, "
            f"max abs err {max_err:.3e})"
        )
        if not ok:
            print(f"  {message}")
            failed.append(name)

        if args.iters > 0:
            times = {
                v: summarize(
                    timeit(
                        fns[v], inputs, args.device, args.warmup, args.iters
                    ),
                    args.stat,
                )
                for v in variants
            }
            rows.append((name, size, times))

    if rows:
        print_table(rows, variants, args, flops)
    if failed:
        raise SystemExit(f"FAILED: {', '.join(failed)}")

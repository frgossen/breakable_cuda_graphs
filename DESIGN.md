# Breakable CUDA Graphs — Design Document

## Overview

`breakable-cuda-graphs` is a pure-Python PyTorch annex package that extends
CUDA graph capture to support **graph breaks** — points where capture is paused,
an eager function runs, and capture resumes into a new graph segment. This
allows CUDA-graph-incompatible operations to be interleaved with captured
segments without abandoning CUDA graphs entirely.

This work was heavily inspired by
[SGLang PR #19102](https://github.com/sgl-project/sglang/pull/19102), which
introduced breakable CUDA graphs for SGLang's model executor.

## API

- **`CUDAGraphSequence`** — the breakable equivalent of `torch.cuda.CUDAGraph`.
  Holds a sequence of `torch.cuda.CUDAGraph` segments and eager callables.
  Provides `replay()`, `reset()`, and `pool()`.
- **`breakable_graph`** — the breakable equivalent of `torch.cuda.graph`.
  Context manager that captures into a `CUDAGraphSequence`. Takes optional
  `pool`, `stream`, `capture_error_mode`, matching the `torch.cuda.graph`
  signature.
- **`@graph_break`** — decorator marking a function as not CUDA-graph-compatible.
  Supports `@graph_break` (bare) and `@graph_break(enable=True/False)`
  (conditional). When called inside a `breakable_graph` capture, the decorator
  pauses capture, runs the function eagerly, inserts a replay closure into the
  sequence, and resumes capture.
- **`force_graph_break()`** — a `@graph_break`-decorated no-op for explicit
  split points.

**Usage.** A `@graph_break`-decorated function pauses capture, runs eagerly, and resumes
capture into a new segment.

```python
@graph_break
def cpu_sync_op(x):
    if x.sum().item() > 0:
        x.clamp_(min=0)

seq = CUDAGraphSequence()
with breakable_graph(seq):
    y = model(x)
    cpu_sync_op(y)
    w = head(y)

seq.replay()
```

`force_graph_break()` inserts an explicit split with no eager work — useful for
debugging or isolating capture regions.

```python
seq2 = CUDAGraphSequence()
with breakable_graph(seq2):
    a = step1(x)
    force_graph_break()
    b = step2(a)

seq2.replay()
```

## Architecture

Instead of capturing into a single `torch.cuda.CUDAGraph`, we capture into a
sequence of them, interleaved with eager callables. `CUDAGraphSequence`
represents this captured sequence as a single interleaved list in `_segments`.
On `replay()`, we iterate the list: graph segments are replayed, callables are
called.

We delegate to `torch.cuda.graph` for each segment rather than calling
`capture_begin`/`capture_end` directly. This means each segment gets the full
`torch.cuda.graph` treatment: stream context management, synchronization, and
error handling.

**Output copying.** An eager `@graph_break` function can return a dynamically allocated buffer,
which may be at a different address on each invocation. CUDA graphs expect the
same addresses for their inputs on replay. To satisfy this constraint, we copy
the result into a known static buffer captured during the initial run, using
`torch.utils._pytree.tree_map` to handle arbitrary nested structures (tuples,
lists, dicts, mixed tensor/non-tensor).

This copy is expensive and almost always avoidable — users can write results to
a pre-allocated buffer instead. We support returned values for convenience but
log a warning when it happens, since it is a performance pitfall on every
replay.

**Context tracking.** A `contextvars.ContextVar` tracks the active `breakable_graph` context. This is
how `@graph_break`-decorated functions know they are inside a capture — they
check the context var to decide whether to pause capture and run eagerly, or
just call through normally. Nested captures are rejected with a clear error.

**Memory pool sharing.** `CUDAGraphSequence` supports both user-provided pools and automatic pool reuse.
All graph segments within a sequence share the same pool — after the first
segment is captured, subsequent segments reuse its pool. Users can also share
pools across sequences via `breakable_graph(seq2, pool=seq1.pool())`.

## Stream Fork/Join Tracking

**The problem.** During CUDA graph capture, work can be forked onto side streams and joined back:

```
                      fork                   join
                      |                      |
main  --[kernel A]----+────[kernel C]────----+----....
                       \                    /
side                    +────[kernel B]----+
```

A fork happens when a side stream waits on the capturing stream
(`side.wait_stream(main)`). A join happens when the capturing stream waits on
the side stream (`main.wait_stream(side)`). These dependencies become edges in
the captured graph.

The issue arises at graph break boundaries. When we call `capture_end()` to seal
a segment, CUDA requires **all** participating side streams to have rejoined.
If a side stream is still forked when a `@graph_break` triggers, `capture_end()`
fails with `cudaStreamCaptureUnjoined`:

```
                      fork    graph break FAILS
                      |       |
main  ──[kernel A]----+──---──+ !! FAILS: side stream not joined
                       \
side                    +────[kernel B]---- still dangling
```

We must auto-join any dangling side streams before ending each segment:

```
                      fork                   auto-join    graph break              new segment
                      |                      |            |                        |
main  ──[kernel A]----+────────────────------+------------+----[eager kernel C]──--+----....
                       \                    /
side                    +──--[kernel B]──--+
```

**Our approach: event hooks.** To auto-join, we need to know which streams are currently forked. We track this
by monkey-patching `torch.cuda.Event.record` and `torch.cuda.Event.wait` at
import time. All higher-level stream synchronization APIs funnel through these:

- `Stream.wait_stream(other)` → `self.wait_event(other.record_event())`
- `Stream.wait_event(event)` → `event.wait(self)`
- `Stream.record_event(event)` → `event.record(self)`

Our `Event.record` hook records which stream recorded each event. Our
`Event.wait` hook checks whether the dependency creates a fork (side stream
depends on capturing stream) or a join (capturing stream depends on side stream),
and updates a `_forked_streams` set accordingly. Before each `capture_end()`, we
iterate over `_forked_streams` and auto-join any remaining side streams.

An advantage of the monkey-patching approach is that it keeps the implementation
separate from PyTorch, which made it easier to prototype. However,
monkey-patching is fragile and we expect to port the basic building blocks for
stream fork/join tracking into PyTorch core sooner rather than later.

Note that there is no way to query CUDA for participating streams or to
intercept stream events at the native CUDA API level. It is always possible to
bypass the PyTorch API and create forks/joins that our tracking is unaware of
(e.g., via direct `cuda-python` bindings). We detect these cases when
`capture_end()` fails with `cudaStreamCaptureUnjoined` and raise an
understandable error.

## Weak References

SGLang [PR #22218](https://github.com/sgl-project/sglang/pull/22218) added a
C++ extension
([`weak_ref_tensor.cpp`](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/memory/weak_ref_tensor.cpp))
that uses PyTorch's
[`at::from_blob`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/ops/from_blob.h)
to weak-ref the args, kwargs, and output tensors in replay closures. This can
save memory but is unsafe and relies on fragile properties of the graphed
programs. A weak-ref tensor shares the same GPU memory without incrementing the
Python storage refcount, so the caching allocator is free to reclaim the memory
once the original tensor is dropped.

**Pool-reuse corruption (verified).** When segments share a memory pool, the pool reuses addresses across segments
that run sequentially — this is the intended memory-saving behavior. With strong
Python references, the storage refcount prevents the pool from reusing an
address that is still referenced. With weak refs, this protection is lost.

We verified this hazard with a concrete test (`test_pool_reuse_with_shared_pool`):

```python
with torch.cuda.graph(g1):
    y = torch.empty(N, device="cuda")
    torch.mul(static_input, 2.0, out=y)         -- y = 10.0 at address A

y_weak = weak_ref_tensor(y)
y = None                                        -- A appears free to pool

with torch.cuda.graph(g2, pool=g1.pool()):
    z = torch.empty(N, device="cuda")           -- reuses address A
    torch.mul(static_input, 3.0, out=z)         -- z = 15.0, overwrites A
    torch.add(z, y_weak, out=output)            -- y_weak reads 15.0, not 10.0
```

On replay: `output = z + y_weak = 15 + 15 = 30`, not the correct `15 + 10 = 25`.
The write to `z` at address A executes before the read of `y_weak` at the same
address, corrupting the result.

With strong refs: the pool cannot reuse A, `z` gets a different address, and
`output = 15 + 10 = 25` — correct. SGLang gets away with this in practice
because their input tensors are pre-allocated static buffers held alive by the
runner object, and typical transformer layers *happen to consume inputs before
allocating large intermediates*. Neither of these is a correctness guarantee.

We use strong references in replay closures. This is always correct regardless
of tensor provenance, usage pattern, or pool sharing configuration. We could
support optional unsafe weak-refing as an opt-in mode for users who accept the
risks.

## CUDA GC and Deterministic Cleanup

This is a pre-existing issue with CUDA graphs in PyTorch, not specific to
breakable graphs. At its core is a tradeoff between slow repeated `gc.collect()`
calls and strict correctness guarantees surrounding CUDA graph lifetimes.

When `CUDAGraph` objects with side stream references are not destroyed before
the next capture, `torch.cuda.synchronize()` (called inside
`torch.cuda.graph.__enter__`) fails with "operation not permitted when stream is
capturing." PyTorch addressed this with
`torch.compiler.config.force_cudagraph_gc`, which triggers `gc.collect()` before
each capture. This was originally the default behavior but was made opt-in
because `gc.collect()` is expensive for back-to-back captures.

We inherit this problem. Users doing multi-stream capture with graph breaks
should be aware that stale `CUDAGraph` objects can interfere with subsequent
captures. Calling `gc.collect()` or `seq.reset()` between capture sessions is
recommended.

## Discussion

- **Port stream fork/join tracking to PyTorch core?** Our monkey-patching of
  `Event.record`/`Event.wait` works but is fragile. Event-level hooks in
  PyTorch core would be cleaner.

  **Proposal:** move this to PyTorch core by
  registering a callback that fires on stream forks and joins. This would let
  breakable CUDA graphs (and other consumers) listen for stream dependencies
  without patching any APIs.

- **Strictly forbid return values from `@graph_break` functions?** Currently we
  support returned tensors via `_copy_output` but log a warning since the copy
  is almost always avoidable.

  **Proposal:** keep the warning as the default
  behavior. Add a strict mode to `breakable_graph` that turns this into an
  error, for users who want to enforce zero-copy graph breaks.

- **Support unsafe weak references in replay closures?** SGLang weak-refs the
  captured args/kwargs/output to let the caching allocator reclaim memory. As
  discussed in the Weak References section, this is fragile — it depends on
  tensors being kept alive by other means and on the graph reading inputs before
  any allocation overwrites them.

  **Proposal:** we will not support this by default. We will support this as an
  unsafe opt-in, clearly marked as such, if it is a blocker for SGLang adoption.

- **Switch to direct `capture_begin`/`capture_end`?** We currently delegate to
  `torch.cuda.graph` for each segment, which handles stream setup and error
  handling for us. Switching to direct `capture_begin`/`capture_end` would give
  finer control over stream management and avoid the per-segment
  `synchronize()` + `empty_cache()` overhead. The tradeoff is more code to
  maintain and less benefit from future `torch.cuda.graph` improvements.

  **Proposal:** keep delegating to `torch.cuda.graph` for simplicity. We can
  revisit if it turns out to be a performance bottleneck, or address it when
  breakable CUDA graphs are incorporated into PyTorch core.

- **`make_breakable_graphed_callables`?** PyTorch's `make_graphed_callables`
  wraps individual modules as graph-captured callables, allowing data-dependent
  control flow between them. We don't have an equivalent — our API captures a
  contiguous region with breaks. A `make_breakable_graphed_callables` that
  returns callable wrappers aware of `@graph_break` would be a natural
  extension.

## Differences from SGLang PR #19102

| Aspect | SGLang PR | Us |
|:---|:---|:---|
| Implementation language | Uses `cuda-python` bindings for direct CUDA runtime calls | Pure Python, delegates to `torch.cuda.graph` |
| Stream hook | Hooks `wait_stream` only | Hooks `Event.record` and `Event.wait` (covers all paths) |
| Hook lifecycle | Install/uninstall with ref counting | Installed at import time |
| Segment storage | Parallel lists (`_segments` + `_break_fns`) | Single interleaved list |
| Output copying | Explicit `__dict__`/dict/tensor handling | `pytree.tree_map` (handles arbitrary structures) |
| Decorator API | `@eager_on_graph(enable=True)` (always requires parens) | `@graph_break` or `@graph_break(enable=True)` |
| Pool sharing | Added post-PR on `main` | Implemented (intra-sequence) |
| Weak-ref tensors | Added post-PR via C++ extension | TODO |
| GPU resource cleanup | Manual `__del__` on raw handles | Delegates to `torch.cuda.CUDAGraph.__del__` |
| `reset()` | Not implemented | Implemented |

## Test Coverage

42 tests covering:

- **Basic capture/replay** — no breaks, drop-in replacement for `torch.cuda.graph`
- **Graph break placement** — start, middle, end, many breaks, force breaks
- **Decorator variants** — enable toggle, user-defined no-ops
- **Mixed operations** — gemm, element-wise, reduction with multiple breaks
- **Return values** — tensor, tuple, list, dict, mixed tensor/non-tensor, args/kwargs
- **Edge cases** — outside capture (no-op), nested capture (rejected), exceptions
- **Lifecycle** — reset and recapture, multiple independent sequences
- **Memory pools** — sharing across sequences, intra-sequence pool sharing
- **Drop-in replacement** — whole network capture, AMP with gradient scaler
- **Fork/join at graph breaks** — no join, partial join, fork-join-fork-again
- **Fork/join at end of capture** — join, no join, partial join
- **Fork/join API variants** — events, wait_stream, mixed

"""Tests for breakable_cuda_graphs."""

import logging

import breakable_cuda_graphs as bcg
import pytest
import torch
from breakable_cuda_graphs import (
    breakable_graph,
    CUDAGraphSequence,
    force_graph_break,
    graph_break,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# Multi-stream tests leave CUDAGraph objects with side stream references.
# When these are GC'd lazily, their stale capture state causes subsequent
# captures to fail. This fixture enables PyTorch's gc.collect() before each
# capture to ensure deterministic cleanup.
@pytest.fixture()
def force_cudagraph_gc():
    old = torch.compiler.config.force_cudagraph_gc
    torch.compiler.config.force_cudagraph_gc = True
    yield
    torch.compiler.config.force_cudagraph_gc = old


# ---------------------------------------------------------------------------
# Basic capture and replay — no graph breaks.
# ---------------------------------------------------------------------------


@requires_cuda
def test_basic_capture_and_replay():
    seq = CUDAGraphSequence()
    static_input = torch.empty(5, device="cuda")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_output = static_input * 2
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        static_output = static_input * 2

    for val in [3.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(static_output, torch.full((5,), val * 2, device="cuda"))


# ---------------------------------------------------------------------------
# Graph break placement — where breaks can occur in a capture.
# ---------------------------------------------------------------------------


@requires_cuda
def test_graph_break_sandwich():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    def step1(dst: torch.Tensor, src: torch.Tensor):
        dst.copy_(src)

    @graph_break
    def step2(x: torch.Tensor):
        x.mul_(3.0)

    def step3(x: torch.Tensor):
        x.add_(1.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        step1(buf, src)
        step2(buf)
        step3(buf)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    assert len(seq._segments) == 3
    assert isinstance(seq._segments[0], torch.cuda.CUDAGraph)
    assert callable(seq._segments[1])
    assert isinstance(seq._segments[2], torch.cuda.CUDAGraph)

    for val in [2.0, 4.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))


@requires_cuda
def test_graph_break_at_start():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    @graph_break
    def step1(dst: torch.Tensor, src: torch.Tensor):
        dst.copy_(src)

    def step2(x: torch.Tensor):
        x.mul_(3.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        step1(buf, src)
        step2(buf)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    assert len(seq._segments) == 3
    assert isinstance(seq._segments[0], torch.cuda.CUDAGraph)
    assert callable(seq._segments[1])
    assert isinstance(seq._segments[2], torch.cuda.CUDAGraph)

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), val * 3.0, device="cuda"))


@requires_cuda
def test_graph_break_at_end():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    def step1(dst: torch.Tensor, src: torch.Tensor):
        dst.copy_(src)
        dst.mul_(3.0)

    @graph_break
    def step2(x: torch.Tensor):
        x.add_(1.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        step1(buf, src)
        step2(buf)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    assert len(seq._segments) == 3
    assert isinstance(seq._segments[0], torch.cuda.CUDAGraph)
    assert callable(seq._segments[1])
    assert isinstance(seq._segments[2], torch.cuda.CUDAGraph)

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))


@requires_cuda
def test_many_breaks_everywhere():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    def step1(dst: torch.Tensor, src: torch.Tensor):
        dst.copy_(src)

    @graph_break
    def step2(x: torch.Tensor):
        x.mul_(3.0)

    def step3(x: torch.Tensor):
        x.add_(1.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        force_graph_break()
        force_graph_break()
        step1(buf, src)
        force_graph_break()
        step2(buf)
        force_graph_break()
        step3(buf)
        force_graph_break()
        force_graph_break()

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    # 6 force_graph_break + 1 @graph_break = 7 breaks
    # each break adds 1 callable + 1 graph, starting with 1 graph: 1 + 7*2 = 15
    assert len(seq._segments) == 15

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))


@graph_break
def _user_defined_noop() -> None:
    pass


@requires_cuda
@pytest.mark.parametrize(
    "break_fn", [force_graph_break, _user_defined_noop], ids=["builtin", "user_noop"]
)
def test_force_graph_break(break_fn):
    buf = torch.empty(5, device="cuda")

    def all_steps(x: torch.Tensor):
        x.fill_(2.0)
        break_fn()
        x.mul_(3.0)
        break_fn()
        x.add_(1.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            all_steps(buf)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf)

    assert len(seq._segments) == 5
    assert isinstance(seq._segments[0], torch.cuda.CUDAGraph)
    assert callable(seq._segments[1])
    assert isinstance(seq._segments[2], torch.cuda.CUDAGraph)
    assert callable(seq._segments[3])
    assert isinstance(seq._segments[4], torch.cuda.CUDAGraph)

    seq.replay()
    assert torch.equal(buf, torch.full((5,), 7.0, device="cuda"))


@requires_cuda
@pytest.mark.parametrize("enable", [True, False])
def test_graph_break_enable_toggle(enable):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    def step1(dst: torch.Tensor, src: torch.Tensor):
        dst.copy_(src)

    @graph_break(enable=enable)
    def step2(x: torch.Tensor):
        x.mul_(3.0)

    def step3(x: torch.Tensor):
        x.add_(1.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        step1(buf, src)
        step2(buf)
        step3(buf)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    if enable:
        assert len(seq._segments) == 3
    else:
        assert len(seq._segments) == 1

    for val in [2.0, 4.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))


@requires_cuda
def test_graph_break_on_bound_method():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    class Scaler:
        def __init__(self, factor: float):
            self.factor = factor

        @graph_break
        def scale(self, x: torch.Tensor):
            x.mul_(self.factor)

    scaler = Scaler(3.0)

    def workload(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        scaler.scale(buf)
        buf.add_(1.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            workload(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        workload(buf, static_input)

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))


# ---------------------------------------------------------------------------
# Mixed operations — gemm, element-wise, reductions with graph breaks.
# ---------------------------------------------------------------------------


@requires_cuda
def test_mixed_ops_multiple_breaks():
    N = 64
    static_a = torch.empty(N, N, device="cuda")
    static_b = torch.empty(N, N, device="cuda")
    buf = torch.empty(N, N, device="cuda")
    scalar = torch.empty(1, device="cuda")

    def gemm():
        buf.copy_(static_a @ static_b)

    @graph_break
    def scale():
        buf.mul_(2.0)

    def bias():
        buf.add_(1.0)

    @graph_break
    def reduce():
        scalar.copy_(buf.sum().unsqueeze(0))

    def postprocess():
        buf.fill_(scalar[0])

    def all_steps():
        gemm()
        scale()
        bias()
        reduce()
        postprocess()

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_a.fill_(0.1)
            static_b.fill_(0.2)
            all_steps()
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps()

    # gemm | scale | bias + reduce | postprocess
    assert len(seq._segments) == 5
    assert isinstance(seq._segments[0], torch.cuda.CUDAGraph)
    assert callable(seq._segments[1])
    assert isinstance(seq._segments[2], torch.cuda.CUDAGraph)
    assert callable(seq._segments[3])
    assert isinstance(seq._segments[4], torch.cuda.CUDAGraph)

    for a_val, b_val in [(0.1, 0.2), (0.5, 0.3)]:
        static_a.fill_(a_val)
        static_b.fill_(b_val)
        seq.replay()
        expected_elem = a_val * b_val * N
        expected_elem = expected_elem * 2.0 + 1.0
        expected_sum = expected_elem * N * N
        assert torch.allclose(scalar, torch.tensor([expected_sum], device="cuda"))


@requires_cuda
def test_user_joined_stream_before_graph_break(force_cudagraph_gc):
    """Side stream forked and joined by the user before a graph break.

    Our event hooks should track the join (discard from _forked_streams),
    so the auto-join at the break point is a no-op. Verifies we don't
    crash by trying to join an already-joined stream.
    """
    buf = torch.empty(5, device="cuda")
    side = torch.cuda.Stream()

    @graph_break
    def eager_step():
        buf.mul_(2.0)

    seq = CUDAGraphSequence()

    # Warmup.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            buf.fill_(1.0)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                buf.add_(1.0)
            torch.cuda.current_stream().wait_stream(side)
            eager_step()
            buf.add_(10.0)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        buf.fill_(1.0)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            buf.add_(1.0)
        torch.cuda.current_stream().wait_stream(side)
        eager_step()
        buf.add_(10.0)

    seq.replay()
    assert torch.equal(buf, torch.full((5,), 14.0, device="cuda"))


# ---------------------------------------------------------------------------
# Return values — graph break functions returning tensors and structures.
# ---------------------------------------------------------------------------


@requires_cuda
def test_graph_break_returns_tensor():
    static_input = torch.empty(5, device="cuda")
    result = torch.empty(5, device="cuda")

    @graph_break
    def compute(x):
        return x * 3.0

    def all_steps():
        tmp = compute(static_input)
        result.copy_(tmp + 1.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps()
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps()

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(result, torch.full((5,), val * 3.0 + 1.0, device="cuda"))


@requires_cuda
@pytest.mark.parametrize(
    "pack,unpack",
    [
        (lambda a, b: [a, b], lambda out: (out[0], out[1])),
        (lambda a, b: (a, b), lambda out: (out[0], out[1])),
        (lambda a, b: {"x": a, "y": b}, lambda out: (out["x"], out["y"])),
        (lambda a, b: {"x": a, "y": b, "s": 1.0}, lambda out: (out["x"], out["y"])),
    ],
    ids=["list", "tuple", "dict", "dict_with_non_tensor"],
)
def test_graph_break_returns_structured(pack, unpack):
    static_input = torch.empty(5, device="cuda")
    result_a = torch.empty(5, device="cuda")
    result_b = torch.empty(5, device="cuda")

    @graph_break
    def compute(x):
        return pack(x * 2.0, x * 3.0)

    def all_steps():
        a, b = unpack(compute(static_input))
        result_a.copy_(a)
        result_b.copy_(b)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps()
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps()

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(result_a, torch.full((5,), val * 2.0, device="cuda"))
        assert torch.equal(result_b, torch.full((5,), val * 3.0, device="cuda"))


@requires_cuda
def test_graph_break_with_args_kwargs():
    static_input = torch.empty(5, device="cuda")
    result_sum = torch.empty(5, device="cuda")

    @graph_break
    def compute(
        *args: torch.Tensor, scale: float, extra: float
    ) -> tuple[torch.Tensor, float]:
        return sum(args) * scale, extra * 2.0

    def all_steps(scale: float, extra: float) -> float:
        a = static_input * 2.0
        b = static_input * 3.0
        tmp, extra = compute(a, b, scale=scale, extra=extra)
        extra *= 3
        result_sum.copy_(tmp)
        return extra

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps(0.5, 0.1)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        extra = all_steps(0.5, 0.1)

    assert extra == 0.1 * 2.0 * 3

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(
            result_sum, torch.full((5,), (val * 2.0 + val * 3.0) * 0.5, device="cuda")
        )


@requires_cuda
def test_graph_break_return_value_warning(caplog):
    static_input = torch.empty(5, device="cuda")
    result = torch.empty(5, device="cuda")

    @graph_break
    def compute(x):
        return x * 3.0

    def all_steps():
        tmp = compute(static_input)
        result.copy_(tmp + 1.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps()
    torch.cuda.current_stream().wait_stream(s)

    with caplog.at_level(logging.WARNING, logger="breakable_cuda_graphs"):
        with breakable_graph(seq):
            all_steps()

    assert any("requires a copy" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Edge cases — outside capture, reset, nesting, exceptions.
# ---------------------------------------------------------------------------


@requires_cuda
def test_graph_break_outside_capture():
    buf = torch.empty(5, device="cuda")

    @graph_break
    def step(x: torch.Tensor):
        x.mul_(3.0)

    buf.fill_(2.0)
    step(buf)
    assert torch.equal(buf, torch.full((5,), 6.0, device="cuda"))

    force_graph_break()

    buf.fill_(4.0)
    step(buf)
    assert torch.equal(buf, torch.full((5,), 12.0, device="cuda"))


@requires_cuda
def test_reset_and_recapture():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    @graph_break
    def eager_mul(x: torch.Tensor):
        x.mul_(3.0)

    def workload_a(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        eager_mul(buf)
        buf.add_(1.0)

    def workload_b(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        eager_mul(buf)
        buf.mul_(2.0)

    seq = CUDAGraphSequence()

    # Capture workload A.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            workload_a(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        workload_a(buf, static_input)

    static_input.fill_(2.0)
    seq.replay()
    assert torch.equal(buf, torch.full((5,), 2.0 * 3.0 + 1.0, device="cuda"))

    # Reset and recapture with workload B.
    seq.reset()
    assert len(seq._segments) == 1

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            workload_b(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        workload_b(buf, static_input)

    static_input.fill_(2.0)
    seq.replay()
    assert torch.equal(buf, torch.full((5,), 2.0 * 3.0 * 2.0, device="cuda"))


@requires_cuda
def test_graph_break_nested_call_stack():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    @graph_break
    def leaf_op(x: torch.Tensor) -> torch.Tensor:
        return x * 3.0

    def inner(x: torch.Tensor) -> torch.Tensor:
        x = leaf_op(x)
        return x + 1.0

    def middle(x: torch.Tensor) -> torch.Tensor:
        return inner(x)

    def outer(buf: torch.Tensor, src: torch.Tensor):
        result = middle(src)
        buf.copy_(result * 2.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            outer(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        outer(buf, static_input)

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(
            buf, torch.full((5,), (val * 3.0 + 1.0) * 2.0, device="cuda")
        )


# ---------------------------------------------------------------------------
# Memory pools — sharing across sequences and within segments.
# ---------------------------------------------------------------------------


@requires_cuda
def test_shared_memory_pool():
    static_in_1 = torch.empty(5, device="cuda")
    static_in_2 = torch.empty(5, device="cuda")
    buf_1 = torch.empty(5, device="cuda")
    buf_2 = torch.empty(5, device="cuda")

    @graph_break
    def eager_scale(x: torch.Tensor):
        x.mul_(2.0)

    def workload_1(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        eager_scale(buf)
        buf.add_(1.0)

    def workload_2(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        eager_scale(buf)
        buf.add_(2.0)

    seq1 = CUDAGraphSequence()
    seq2 = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_in_1.fill_(10.0)
            static_in_2.fill_(20.0)
            workload_1(buf_1, static_in_1)
            workload_2(buf_2, static_in_2)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq1):
        workload_1(buf_1, static_in_1)

    with breakable_graph(seq2, pool=seq1.pool()):
        workload_2(buf_2, static_in_2)

    assert seq1.pool() == seq2.pool()

    for v1, v2 in [(10.0, 20.0), (3.0, 7.0)]:
        static_in_1.fill_(v1)
        static_in_2.fill_(v2)
        seq1.replay()
        seq2.replay()
        assert torch.equal(buf_1, torch.full((5,), v1 * 2.0 + 1.0, device="cuda"))
        assert torch.equal(buf_2, torch.full((5,), v2 * 2.0 + 2.0, device="cuda"))


@requires_cuda
def test_intra_sequence_pool_sharing():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    @graph_break
    def eager_step(x: torch.Tensor):
        x.mul_(3.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        eager_step(buf)
        buf.add_(1.0)
        eager_step(buf)
        buf.add_(2.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    graphs = [s for s in seq._segments if isinstance(s, torch.cuda.CUDAGraph)]
    assert len(graphs) == 3
    assert graphs[0].pool() == graphs[1].pool()
    assert graphs[1].pool() == graphs[2].pool()


@requires_cuda
def test_pool_valid_after_reset():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    @graph_break
    def eager_step(x: torch.Tensor):
        x.mul_(3.0)

    def workload(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        eager_step(buf)
        buf.add_(1.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            workload(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        workload(buf, static_input)

    pool_before = seq.pool()
    seq.reset()

    # After reset, pool() fails because the fresh CUDAGraph hasn't been captured yet.
    with pytest.raises(RuntimeError, match="without a preceding successful capture"):
        seq.pool()

    # Recapture using the pool from before reset.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            workload(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq, pool=pool_before):
        workload(buf, static_input)

    assert seq.pool() == pool_before

    static_input.fill_(4.0)
    seq.replay()
    assert torch.equal(buf, torch.full((5,), 4.0 * 3.0 + 1.0, device="cuda"))


@requires_cuda
def test_multiple_captures_in_sequence():
    static_input = torch.empty(5, device="cuda")
    buf_1 = torch.empty(5, device="cuda")
    buf_2 = torch.empty(5, device="cuda")

    @graph_break
    def eager_scale(x: torch.Tensor, factor: float):
        x.mul_(factor)

    def workload_1(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        eager_scale(buf, 3.0)

    def workload_2(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        eager_scale(buf, 5.0)

    seq_1 = CUDAGraphSequence()
    seq_2 = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            workload_1(buf_1, static_input)
            workload_2(buf_2, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq_1):
        workload_1(buf_1, static_input)

    with breakable_graph(seq_2):
        workload_2(buf_2, static_input)

    for val in [2.0, 7.0]:
        static_input.fill_(val)
        seq_1.replay()
        assert torch.equal(buf_1, torch.full((5,), val * 3.0, device="cuda"))
        seq_2.replay()
        assert torch.equal(buf_2, torch.full((5,), val * 5.0, device="cuda"))


@requires_cuda
def test_exception_during_capture():
    buf = torch.empty(5, device="cuda")

    @graph_break
    def failing_step(x: torch.Tensor):
        raise RuntimeError("intentional failure")

    def all_steps(buf: torch.Tensor):
        buf.fill_(2.0)
        failing_step(buf)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            buf.fill_(2.0)
    torch.cuda.current_stream().wait_stream(s)

    with pytest.raises(RuntimeError, match="intentional failure"):
        with breakable_graph(seq):
            all_steps(buf)

    assert bcg._current_breakable_graph_ctx.get() is None


@requires_cuda
def test_replay_after_partial_capture_failure():
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")

    @graph_break
    def failing_step(x: torch.Tensor):
        raise RuntimeError("intentional failure")

    @graph_break
    def eager_step(x: torch.Tensor):
        x.mul_(3.0)

    seq = CUDAGraphSequence()

    # First capture fails mid-way.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            buf.copy_(static_input)
    torch.cuda.current_stream().wait_stream(s)

    with pytest.raises(RuntimeError, match="intentional failure"):
        with breakable_graph(seq):
            buf.copy_(static_input)
            failing_step(buf)

    # Reset and recapture with a working workload.
    seq.reset()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(2.0)
            buf.copy_(static_input)
            eager_step(buf)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        buf.copy_(static_input)
        eager_step(buf)

    for val in [2.0, 5.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), val * 3.0, device="cuda"))


@requires_cuda
def test_nested_breakable_graph_raises():
    buf = torch.empty(5, device="cuda")
    seq1 = CUDAGraphSequence()
    seq2 = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            buf.fill_(2.0)
    torch.cuda.current_stream().wait_stream(s)

    with pytest.raises(
        RuntimeError, match="nested breakable_graph captures are not supported"
    ):
        with breakable_graph(seq1):
            buf.fill_(2.0)
            with breakable_graph(seq2):
                buf.mul_(3.0)


@requires_cuda
def test_exception_during_resume_capture():
    buf = torch.empty(5, device="cuda")

    call_count = 0

    @graph_break
    def eager_step(x: torch.Tensor):
        nonlocal call_count
        call_count += 1
        x.mul_(2.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            buf.fill_(1.0)
            eager_step(buf)
    torch.cuda.current_stream().wait_stream(s)

    original_resume = bcg.breakable_graph._resume_capture

    def failing_resume(self):
        raise RuntimeError("simulated _resume_capture failure")

    bcg.breakable_graph._resume_capture = failing_resume
    try:
        with pytest.raises(RuntimeError, match="simulated _resume_capture failure"):
            with breakable_graph(seq):
                buf.fill_(1.0)
                eager_step(buf)
    finally:
        bcg.breakable_graph._resume_capture = original_resume

    assert bcg._current_breakable_graph_ctx.get() is None


# ---------------------------------------------------------------------------
# Drop-in replacement — breakable_graph used in place of torch.cuda.graph.
# ---------------------------------------------------------------------------


@requires_cuda
def test_whole_network_capture_drop_in():
    N, D_in, H, D_out = 640, 4096, 2048, 1024
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, D_out),
        torch.nn.Dropout(p=0.1),
    ).cuda()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    static_input = torch.randn(N, D_in, device="cuda")
    static_target = torch.randn(N, D_out, device="cuda")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(static_input)
            loss = loss_fn(y_pred, static_target)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    seq = CUDAGraphSequence()
    optimizer.zero_grad(set_to_none=True)
    with breakable_graph(seq):
        static_y_pred = model(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()

    real_inputs = [torch.rand_like(static_input) for _ in range(10)]
    real_targets = [torch.rand_like(static_target) for _ in range(10)]

    for data, target in zip(real_inputs, real_targets):
        static_input.copy_(data)
        static_target.copy_(target)
        seq.replay()

    assert static_y_pred.shape == (N, D_out)
    assert static_loss.ndim == 0


@requires_cuda
def test_amp_with_graph_capture_drop_in():
    N, D_in, H, D_out = 640, 4096, 2048, 1024
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, D_out),
        torch.nn.Dropout(p=0.1),
    ).cuda()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.GradScaler()

    static_input = torch.randn(N, D_in, device="cuda")
    static_target = torch.randn(N, D_out, device="cuda")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                y_pred = model(static_input)
                loss = loss_fn(y_pred, static_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.current_stream().wait_stream(s)

    seq = CUDAGraphSequence()
    optimizer.zero_grad(set_to_none=True)
    with breakable_graph(seq):
        with torch.amp.autocast("cuda"):
            static_y_pred = model(static_input)
            static_loss = loss_fn(static_y_pred, static_target)
        scaler.scale(static_loss).backward()

    real_inputs = [torch.rand_like(static_input) for _ in range(10)]
    real_targets = [torch.rand_like(static_target) for _ in range(10)]

    for data, target in zip(real_inputs, real_targets):
        static_input.copy_(data)
        static_target.copy_(target)
        seq.replay()
        scaler.step(optimizer)
        scaler.update()

    assert static_y_pred.shape == (N, D_out)
    assert static_loss.ndim == 0


# ---------------------------------------------------------------------------
# Fork/join — at graph break boundaries.
# ---------------------------------------------------------------------------


@requires_cuda
def test_fork_no_join_before_graph_break(force_cudagraph_gc):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")
    side = torch.cuda.Stream()

    @graph_break
    def eager_step(x: torch.Tensor):
        x.mul_(2.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        side.wait_stream(torch.cuda.current_stream())  # fork
        with torch.cuda.stream(side):
            buf.add_(1.0)
        # No join — auto-join in _pause_capture must handle it.
        eager_step(buf)
        buf.add_(10.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(1.0)
            all_steps(buf, static_input)
            torch.cuda.current_stream().wait_stream(side)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    for val in [1.0, 3.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(
            buf, torch.full((5,), (val + 1.0) * 2.0 + 10.0, device="cuda")
        )


@requires_cuda
def test_partial_join_before_graph_break(force_cudagraph_gc):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")
    side1 = torch.cuda.Stream()
    side2 = torch.cuda.Stream()

    @graph_break
    def eager_step(x: torch.Tensor):
        x.mul_(2.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        side1.wait_stream(torch.cuda.current_stream())  # fork side1
        side2.wait_stream(torch.cuda.current_stream())  # fork side2
        with torch.cuda.stream(side1):
            buf.add_(1.0)
        with torch.cuda.stream(side2):
            buf.add_(2.0)
        torch.cuda.current_stream().wait_stream(side1)  # join side1 only
        # side2 left for auto-join in _pause_capture.
        eager_step(buf)
        buf.add_(10.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(1.0)
            all_steps(buf, static_input)
            torch.cuda.current_stream().wait_stream(side2)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    for val in [1.0, 3.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(
            buf, torch.full((5,), (val + 1.0 + 2.0) * 2.0 + 10.0, device="cuda")
        )


@requires_cuda
def test_fork_join_fork_again_before_graph_break(force_cudagraph_gc):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")
    side = torch.cuda.Stream()

    @graph_break
    def eager_step(x: torch.Tensor):
        x.mul_(2.0)

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        side.wait_stream(torch.cuda.current_stream())  # fork
        with torch.cuda.stream(side):
            buf.add_(1.0)
        torch.cuda.current_stream().wait_stream(side)  # join
        side.wait_stream(torch.cuda.current_stream())  # fork again
        with torch.cuda.stream(side):
            buf.add_(2.0)
        # No join — auto-join must handle it.
        eager_step(buf)
        buf.add_(10.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(1.0)
            all_steps(buf, static_input)
            torch.cuda.current_stream().wait_stream(side)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    for val in [1.0, 3.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(
            buf, torch.full((5,), (val + 1.0 + 2.0) * 2.0 + 10.0, device="cuda")
        )


# ---------------------------------------------------------------------------
# Fork/join — at end of capture.
# ---------------------------------------------------------------------------


@requires_cuda
def test_fork_join_before_end_of_capture(force_cudagraph_gc):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")
    side = torch.cuda.Stream()

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        side.wait_stream(torch.cuda.current_stream())  # fork
        with torch.cuda.stream(side):
            buf.add_(1.0)
        torch.cuda.current_stream().wait_stream(side)  # join
        buf.mul_(2.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(3.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    for val in [3.0, 7.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), (val + 1.0) * 2.0, device="cuda"))


@requires_cuda
def test_fork_no_join_before_end_of_capture(force_cudagraph_gc):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")
    side = torch.cuda.Stream()

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        side.wait_stream(torch.cuda.current_stream())  # fork
        with torch.cuda.stream(side):
            buf.add_(1.0)
        # No join — auto-join in __exit__ must handle it.
        buf.mul_(2.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(3.0)
            all_steps(buf, static_input)
            torch.cuda.current_stream().wait_stream(side)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    for val in [3.0, 7.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), (val + 1.0) * 2.0, device="cuda"))


@requires_cuda
def test_partial_join_before_end_of_capture(force_cudagraph_gc):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")
    side1 = torch.cuda.Stream()
    side2 = torch.cuda.Stream()

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        side1.wait_stream(torch.cuda.current_stream())  # fork side1
        side2.wait_stream(torch.cuda.current_stream())  # fork side2
        with torch.cuda.stream(side1):
            buf.add_(1.0)
        with torch.cuda.stream(side2):
            buf.add_(2.0)
        torch.cuda.current_stream().wait_stream(side1)  # join side1 only
        # side2 left for auto-join in __exit__.
        buf.mul_(2.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(3.0)
            all_steps(buf, static_input)
            torch.cuda.current_stream().wait_stream(side2)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    for val in [3.0, 7.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(
            buf, torch.full((5,), (val + 1.0 + 2.0) * 2.0, device="cuda")
        )


# ---------------------------------------------------------------------------
# Fork/join — API variants (events vs wait_stream).
# ---------------------------------------------------------------------------


@requires_cuda
def test_fork_join_via_events(force_cudagraph_gc):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")
    side = torch.cuda.Stream()

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        e = torch.cuda.Event()
        e.record(torch.cuda.current_stream())
        e.wait(side)  # fork via event
        with torch.cuda.stream(side):
            buf.add_(1.0)
        e2 = torch.cuda.Event()
        e2.record(side)
        e2.wait(torch.cuda.current_stream())  # join via event
        buf.mul_(2.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(3.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    for val in [3.0, 7.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), (val + 1.0) * 2.0, device="cuda"))


@requires_cuda
def test_fork_join_mixed_events_and_wait_stream(force_cudagraph_gc):
    static_input = torch.empty(5, device="cuda")
    buf = torch.empty(5, device="cuda")
    side = torch.cuda.Stream()

    def all_steps(buf: torch.Tensor, src: torch.Tensor):
        buf.copy_(src)
        e = torch.cuda.Event()
        e.record(torch.cuda.current_stream())
        e.wait(side)  # fork via event
        with torch.cuda.stream(side):
            buf.add_(1.0)
        torch.cuda.current_stream().wait_stream(side)  # join via wait_stream
        buf.mul_(2.0)

    seq = CUDAGraphSequence()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_input.fill_(3.0)
            all_steps(buf, static_input)
    torch.cuda.current_stream().wait_stream(s)

    with breakable_graph(seq):
        all_steps(buf, static_input)

    for val in [3.0, 7.0]:
        static_input.fill_(val)
        seq.replay()
        assert torch.equal(buf, torch.full((5,), (val + 1.0) * 2.0, device="cuda"))


# ---------------------------------------------------------------------------
# Concurrent captures — two threads capturing with graph breaks simultaneously.
# ---------------------------------------------------------------------------


@requires_cuda
def test_concurrent_captures_with_graph_breaks():
    import threading

    barrier = threading.Barrier(2)
    results = {}
    errors = {}

    def capture_and_replay(name: str, multiplier: float, stream: torch.cuda.Stream):
        try:
            static_input = torch.empty(5, device="cuda")
            buf = torch.empty(5, device="cuda")

            @graph_break
            def eager_step(x: torch.Tensor, factor: float):
                x.mul_(factor)

            def all_steps(buf: torch.Tensor, src: torch.Tensor):
                buf.copy_(src)
                eager_step(buf, multiplier)

            seq = CUDAGraphSequence()

            with torch.cuda.stream(stream):
                for _ in range(3):
                    static_input.fill_(2.0)
                    all_steps(buf, static_input)
            torch.cuda.synchronize()

            barrier.wait()

            with breakable_graph(seq, stream=stream, capture_error_mode="thread_local"):
                all_steps(buf, static_input)

            static_input.fill_(4.0)
            seq.replay()
            torch.cuda.synchronize()
            results[name] = buf.clone()
        except Exception as e:
            errors[name] = e

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    t1 = threading.Thread(target=capture_and_replay, args=("t1", 3.0, s1))
    t2 = threading.Thread(target=capture_and_replay, args=("t2", 5.0, s2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Thread errors: {errors}"
    assert torch.equal(results["t1"], torch.full((5,), 12.0, device="cuda"))
    assert torch.equal(results["t2"], torch.full((5,), 20.0, device="cuda"))


# ---------------------------------------------------------------------------
# Weak reference hazard — verifies strong refs prevent pool-reuse corruption.
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("use_weak_ref", [False, True], ids=["strong_ref", "weak_ref"])
def test_pool_reuse_with_shared_pool(use_weak_ref):
    """With strong refs, the pool cannot reuse an address that is still
    referenced — result is correct.  With weak refs (as SGLang does via
    PR #22218), the pool reuses the address and a subsequent write overwrites
    the data before it is read — confirmed corruption.
    """
    if use_weak_ref:
        from textwrap import dedent

        from torch.utils.cpp_extension import load_inline

        ext = load_inline(
            name="weak_ref_hazard_test",
            cpp_sources=dedent(
                """\
                #include <ATen/ATen.h>
                at::Tensor weak_ref_tensor(const at::Tensor& tensor) {
                    return at::from_blob(
                        tensor.data_ptr(), tensor.sizes().vec(),
                        tensor.strides().vec(), tensor.options());
                }
            """
            ),
            functions=["weak_ref_tensor"],
            verbose=False,
        )

    N = 512
    static_input = torch.full((N,), 5.0, device="cuda")
    output = torch.empty(N, device="cuda")
    g1 = torch.cuda.CUDAGraph()
    g2 = torch.cuda.CUDAGraph()

    # Segment 1: y = input * 2 (y = 10.0 at address A)
    with torch.cuda.graph(g1):
        y = torch.empty(N, device="cuda")
        torch.mul(static_input, 2.0, out=y)

    y_ptr = y.data_ptr()
    if use_weak_ref:
        y_ref = ext.weak_ref_tensor(y)
        y = None  # drop strong ref — address A appears free to pool
    else:
        y_ref = y  # strong ref — pool cannot reuse address A

    # Segment 2: allocates z (may reuse address A), then reads y_ref
    with torch.cuda.graph(g2, pool=g1.pool()):
        z = torch.empty(N, device="cuda")  # with weak ref: falsely reuses address A
        torch.mul(static_input, 3.0, out=z)  # z = 15.0, overwrites A
        torch.add(z, y_ref, out=output)  # output = z + y_ref

    # Replay
    static_input.fill_(5.0)
    g1.replay()
    g2.replay()
    torch.cuda.synchronize()

    correct = 15.0 + 10.0  # 25.0 — z + y with correct y data
    corrupted = 15.0 + 15.0  # 30.0 — z + z because y was overwritten

    if use_weak_ref:
        assert y_ptr == z.data_ptr(), "Pool must reuse y's address"
        assert torch.equal(output, torch.full((N,), corrupted, device="cuda"))
    else:
        assert y_ptr != z.data_ptr(), "Pool must not reuse y's address"
        assert torch.equal(output, torch.full((N,), correct, device="cuda"))

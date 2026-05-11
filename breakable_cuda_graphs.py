"""Breakable CUDA graphs for PyTorch."""

from __future__ import annotations

import contextvars
import functools
import logging
import weakref
from typing import Callable

import torch

__version__ = "0.0.0"
__all__ = [
    "CUDAGraphSequence",
    "breakable_graph",
    "graph_break",
    "force_graph_break",
]

log = logging.getLogger(__name__)

_current_breakable_graph_ctx: contextvars.ContextVar[breakable_graph | None] = (
    contextvars.ContextVar("current_breakable_graph_ctx", default=None)
)


def _capturing_event_record(
    self: torch.cuda.Event, stream: torch.cuda.Stream | None = None
):
    if stream is None:
        stream = torch.cuda.current_stream()
    ctx = _current_breakable_graph_ctx.get()
    if ctx is not None:
        ctx._event_to_stream[self] = stream
    return _base_event_record(self, stream)


def _capturing_event_wait(
    self: torch.cuda.Event, stream: torch.cuda.Stream | None = None
):
    if stream is None:
        stream = torch.cuda.current_stream()
    ctx = _current_breakable_graph_ctx.get()
    if ctx is None:
        return _base_event_wait(self, stream)

    recording_stream = ctx._event_to_stream.get(self)
    if recording_stream is None or ctx._is_capturing_stream(
        recording_stream
    ) == ctx._is_capturing_stream(stream):
        return _base_event_wait(self, stream)

    if ctx._is_capturing_stream(recording_stream):
        ctx._forked_streams.add(stream)
    else:
        ctx._forked_streams.discard(recording_stream)

    return _base_event_wait(self, stream)


# Hooking Event.record and Event.wait is sufficient to track all stream
# fork/join dependencies. The higher-level APIs all funnel through these:
#   Stream.wait_stream(other)    -> self.wait_event(other.record_event())
#   Stream.wait_event(event)     -> event.wait(self)
#   Stream.record_event(event)   -> event.record(self)
_base_event_record = torch.cuda.Event.record
_base_event_wait = torch.cuda.Event.wait
torch.cuda.Event.record = _capturing_event_record  # type: ignore[assignment]
torch.cuda.Event.wait = _capturing_event_wait  # type: ignore[assignment]


def _copy_leaf(dst, src):
    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)
        return dst
    return src


def graph_break(fn: Callable | None = None, *, enable: bool = True) -> Callable:
    """Mark a function as not CUDA-graph-compatible.

    During :class:`breakable_graph` capture, calls to decorated functions will
    break the graph and run eagerly.

    Can be used as ``@graph_break`` or ``@graph_break(enable=True)``.
    """

    def decorator(fn: Callable) -> Callable:
        if not enable:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = _current_breakable_graph_ctx.get()
            if ctx is None:
                return fn(*args, **kwargs)

            ctx._pause_capture()

            captured_result = fn(*args, **kwargs)
            if captured_result is not None:
                log.warning(
                    "%s returns a value; this requires a copy on every replay, "
                    "which may hurt performance. Prefer writing results to a "
                    "pre-allocated buffer instead.",
                    fn.__qualname__,
                )

            def replay_fn():
                result = fn(*args, **kwargs)
                if captured_result is not None:
                    torch.utils._pytree.tree_map(_copy_leaf, captured_result, result)
                    return captured_result
                else:
                    assert result is None

            ctx._insert_eager(replay_fn)
            ctx._resume_capture()

            return captured_result

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


@graph_break
def force_graph_break() -> None:
    pass


class CUDAGraphSequence:
    """A replayable sequence of CUDA graph segments and eager callables.

    Built by :class:`breakable_graph` during capture.  Call :meth:`replay` to
    re-execute the entire sequence.
    """

    def __init__(self) -> None:
        self._segments: list[torch.cuda.CUDAGraph | Callable] = []
        self._append_graph()

    def _append_graph(self) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        self._segments.append(g)
        return g

    def _append_eager(self, fn: Callable) -> None:
        self._segments.append(fn)

    def pool(self):
        return self._segments[0].pool()

    def _current_graph(self) -> torch.cuda.CUDAGraph:
        g = self._segments[-1]
        assert isinstance(g, torch.cuda.CUDAGraph)
        return g

    def replay(self) -> None:
        for segment in self._segments:
            if isinstance(segment, torch.cuda.CUDAGraph):
                segment.replay()
                continue
            assert callable(segment)
            segment()

    def reset(self) -> None:
        for segment in self._segments:
            if isinstance(segment, torch.cuda.CUDAGraph):
                segment.reset()
        self._segments.clear()
        self._append_graph()


class breakable_graph:
    """Like :class:`torch.cuda.graph` but allows :func:`graph_break`-decorated
    functions to break the capture into multiple segments.

    Delegates to :class:`torch.cuda.graph` for each segment. For concurrent
    captures from multiple threads with graph breaks, use ``thread_local``
    capture error mode and separate streams per thread.
    """

    def __init__(
        self,
        cuda_graph_sequence: CUDAGraphSequence,
        pool=None,
        stream: torch.cuda.Stream | None = None,
        capture_error_mode: str = "global",
    ) -> None:
        self._seq = cuda_graph_sequence
        self._pool = pool
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._new_graph_ctx(cuda_graph_sequence._current_graph())

    def _new_graph_ctx(self, g: torch.cuda.CUDAGraph):
        pool = self._pool if len(self._seq._segments) <= 1 else self._seq.pool()
        self._graph_ctx = torch.cuda.graph(
            g,
            pool=pool,
            stream=self._stream,
            capture_error_mode=self._capture_error_mode,
        )

    def _is_capturing_stream(self, s: torch.cuda.Stream) -> bool:
        return (
            s is self._capturing_stream
            or s.cuda_stream == self._capturing_stream.cuda_stream
        )

    def _insert_eager(self, fn: Callable) -> None:
        assert not self._is_capturing()
        self._seq._append_eager(fn)

    def _is_capturing(self) -> bool:
        return self._graph_ctx is not None

    def _join_forked_streams(self):
        while self._forked_streams:
            stream = next(iter(self._forked_streams))
            self._capturing_stream.wait_stream(stream)

    def _pause_capture(self):
        self._join_forked_streams()
        self._graph_ctx.__exit__(None, None, None)
        self._graph_ctx = None
        self._event_to_stream.clear()

    def _resume_capture(self):
        g = self._seq._append_graph()
        self._new_graph_ctx(g)
        self._graph_ctx.__enter__()

    def __enter__(self):
        if _current_breakable_graph_ctx.get() is not None:
            raise RuntimeError("nested breakable_graph captures are not supported")
        self._forked_streams: set[torch.cuda.Stream] = set()
        self._event_to_stream: weakref.WeakKeyDictionary[
            torch.cuda.Event, torch.cuda.Stream
        ] = weakref.WeakKeyDictionary()
        self._graph_ctx.__enter__()
        # Must read current_stream after __enter__, which switches to the capture stream.
        self._capturing_stream = torch.cuda.current_stream()
        self._token = _current_breakable_graph_ctx.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._join_forked_streams()
        # _graph_ctx is None if an exception occurred during a graph break
        # (after _pause_capture but before _resume_capture).
        if self._graph_ctx is not None:
            self._graph_ctx.__exit__(exc_type, exc_val, exc_tb)
        self._event_to_stream.clear()
        _current_breakable_graph_ctx.reset(self._token)
        return False

"""Reference tests for plain torch.cuda CUDA graphs.

These mirror the examples from the PyTorch documentation at
https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs
"""

from itertools import chain

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@requires_cuda
def test_basic_capture_and_replay():
    g = torch.cuda.CUDAGraph()
    static_input = torch.empty((5,), device="cuda")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_output = static_input * 2
    torch.cuda.current_stream().wait_stream(s)

    with torch.cuda.graph(g):
        static_output = static_input * 2

    static_input.copy_(torch.full((5,), 3.0, device="cuda"))
    g.replay()
    assert torch.equal(static_output, torch.full((5,), 6.0, device="cuda"))

    static_input.copy_(torch.full((5,), 4.0, device="cuda"))
    g.replay()
    assert torch.equal(static_output, torch.full((5,), 8.0, device="cuda"))


@requires_cuda
def test_whole_network_capture():
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

    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        static_y_pred = model(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()

    real_inputs = [torch.rand_like(static_input) for _ in range(10)]
    real_targets = [torch.rand_like(static_target) for _ in range(10)]

    for data, target in zip(real_inputs, real_targets):
        static_input.copy_(data)
        static_target.copy_(target)
        g.replay()

    assert static_y_pred.shape == (N, D_out)
    assert static_loss.ndim == 0


@requires_cuda
def test_partial_network_capture():
    N, D_in, H, D_out = 640, 4096, 2048, 1024

    module1 = torch.nn.Linear(D_in, H).cuda()
    module2 = torch.nn.Linear(H, D_out).cuda()
    module3 = torch.nn.Linear(H, D_out).cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        chain(
            module1.parameters(),
            module2.parameters(),
            module3.parameters(),
        ),
        lr=0.1,
    )

    x = torch.randn(N, D_in, device="cuda")
    h = torch.randn(N, H, device="cuda", requires_grad=True)

    module1 = torch.cuda.make_graphed_callables(module1, (x,))
    module2 = torch.cuda.make_graphed_callables(module2, (h,))
    module3 = torch.cuda.make_graphed_callables(module3, (h,))

    real_inputs = [torch.rand_like(x) for _ in range(10)]
    real_targets = [torch.randn(N, D_out, device="cuda") for _ in range(10)]

    for data, target in zip(real_inputs, real_targets):
        optimizer.zero_grad(set_to_none=True)
        tmp = module1(data)
        if tmp.sum().item() > 0:
            tmp = module2(tmp)
        else:
            tmp = module3(tmp)
        loss = loss_fn(tmp, target)
        loss.backward()
        optimizer.step()


@requires_cuda
def test_amp_with_graph_capture():
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
        for i in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                y_pred = model(static_input)
                loss = loss_fn(y_pred, static_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        with torch.amp.autocast("cuda"):
            static_y_pred = model(static_input)
            static_loss = loss_fn(static_y_pred, static_target)
        scaler.scale(static_loss).backward()

    real_inputs = [torch.rand_like(static_input) for _ in range(10)]
    real_targets = [torch.rand_like(static_target) for _ in range(10)]

    for data, target in zip(real_inputs, real_targets):
        static_input.copy_(data)
        static_target.copy_(target)
        g.replay()
        scaler.step(optimizer)
        scaler.update()

    assert static_y_pred.shape == (N, D_out)
    assert static_loss.ndim == 0


@requires_cuda
def test_multi_stream_capture():
    g = torch.cuda.CUDAGraph()
    static_input = torch.empty(5, device="cuda")

    s = torch.cuda.Stream()

    # Warmup.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            static_output = static_input * 2
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                static_output = static_output + 1
            torch.cuda.current_stream().wait_stream(s)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    with torch.cuda.graph(g):
        static_output = static_input * 2
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            static_output = static_output + 1
        torch.cuda.current_stream().wait_stream(s)

    static_input.copy_(torch.full((5,), 3.0, device="cuda"))
    g.replay()
    assert torch.equal(static_output, torch.full((5,), 7.0, device="cuda"))


@requires_cuda
def test_memory_pool_sharing():
    g1 = torch.cuda.CUDAGraph()
    g2 = torch.cuda.CUDAGraph()
    static_in_1 = torch.empty(5, device="cuda")
    static_in_2 = torch.empty(5, device="cuda")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_out_1 = static_in_1 + 1
            static_out_2 = static_in_2 + 2
    torch.cuda.current_stream().wait_stream(s)

    with torch.cuda.graph(g1):
        static_out_1 = static_in_1 + 1

    with torch.cuda.graph(g2, pool=g1.pool()):
        static_out_2 = static_in_2 + 2

    static_in_1.fill_(10.0)
    static_in_2.fill_(20.0)
    g1.replay()
    g2.replay()
    assert torch.equal(static_out_1, torch.full((5,), 11.0, device="cuda"))
    assert torch.equal(static_out_2, torch.full((5,), 22.0, device="cuda"))

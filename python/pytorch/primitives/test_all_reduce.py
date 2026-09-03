import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_all_reduce(rank, world_size):
    # Use a fresh, different port to avoid the "address already in use" conflict
    # You can also use a string like "tcp://localhost:23456"
    init_method = "tcp://127.0.0.1:28500"

    # Initialize by explicitly passing the init_method, rank, and world_size
    dist.init_process_group(
        backend="gloo", init_method=init_method, rank=rank, world_size=world_size
    )

    # Each process initializes a tensor
    tensor = torch.ones(2, 2)

    # Execute all_reduce across the 2 simulated processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Since world_size = 2, the resulting values must be 2.0 (1.0 + 1.0)
    assert torch.equal(tensor, torch.ones(2, 2) * world_size)

    dist.destroy_process_group()


def test_all_reduce_multi_process():
    world_size = 2

    # mp.spawn launches the 2 processes cleanly
    mp.spawn(run_all_reduce, args=(world_size,), nprocs=world_size, join=True)  # type: ignore

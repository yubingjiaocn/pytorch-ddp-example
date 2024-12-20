import argparse
import os
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, device = cuda:{torch.cuda.current_device()} \n", end=''
    )

    model = ToyModel().cuda()
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10).cuda())
    labels = torch.randn(20, 5).cuda()
    loss_fn(outputs, labels).backward()
    optimizer.step()


def spmd_main():
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This example requires GPU support.")

    # Ensure we have all required environment variables
    required_env_vars = ["LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_vars:
        raise RuntimeError(
            f"Missing required environment variables: {missing_vars}\n"
            "Please use torchrun to launch this script."
        )

    # Set the device at the beginning of the process
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"Local rank {local_rank} is greater than available GPUs ({torch.cuda.device_count()})"
        )

    torch.cuda.set_device(local_rank)

    # These are the parameters used to initialize the process group
    env_dict = {key: os.environ[key] for key in required_env_vars}
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    # Initialize process group with NCCL backend for GPU training
    dist.init_process_group(backend="nccl")

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )

    demo_basic()

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    # When using torchrun, we don't need command line arguments
    # as everything is set via environment variables
    spmd_main()

# PyTorch Distributed Data Parallel (DDP) Example

This repository contains examples of using PyTorch's Distributed Data Parallel (DDP) for distributed training. The code is modified from the [official PyTorch examples](https://github.com/pytorch/examples/blob/main/distributed/ddp/example.py) with additional containerization and orchestration support.

## Project Structure

- `example.py`: A simplified DDP example that demonstrates basic distributed training using NCCL backend. This version is designed to work with `torchrun` and includes proper environment validation and GPU device management.
- `main.py`: Contains three different DDP examples:
  1. Basic DDP training
  2. DDP with checkpointing
  3. DDP with model parallelism
- `Dockerfile`: Configures a container with CUDA 12.2 and PyTorch 2.5.0, setting up the necessary environment for distributed training.
- `serve.sh`: Shell script that launches the distributed training using `torchrun`. It automatically detects available GPUs and configures the distributed environment.
- `job.yaml`: Kubernetes Job configuration for running the distributed training in a cluster environment.

## Requirements

- CUDA-capable GPUs
- NVIDIA Container Runtime (for Docker)
- Kubernetes cluster with GPU support (for deployment)

## Docker Container

The Dockerfile sets up an environment with:
- Ubuntu 22.04 base with CUDA 12.2 runtime
- Python 3.10
- PyTorch 2.5.0 with CUDA 12.1 support
- Required system dependencies

## Running the Examples

### Local Development

To run the example locally with multiple GPUs:

```bash
# Run the simplified example
./serve.sh

# Or run the main examples directly
python main.py
```

Note: The main examples require at least 4 GPUs to run all demonstrations.

### Docker Container

Build and run the container:

```bash
docker build -t pytorch-ddp .
docker run --gpus all pytorch-ddp
```

### Kubernetes Deployment

Deploy to a Kubernetes cluster:

```bash
kubectl apply -f job.yaml
```

The Kubernetes job configuration:
- Requests 4 GPUs
- Uses shared memory for inter-process communication
- Configures GPU tolerations and node selection
- Sets up a 1GB shared memory volume for efficient communication

## Implementation Details

### Distributed Training

The examples demonstrate different aspects of PyTorch's distributed training capabilities:

1. Basic DDP Example:
   - Initializes process groups
   - Moves model to GPU
   - Performs distributed training with synchronized gradients

2. Checkpoint Example (in main.py):
   - Shows how to save and load model checkpoints in a distributed setting
   - Uses barriers to ensure proper synchronization between processes

3. Model Parallel Example (in main.py):
   - Demonstrates combining model parallelism with data parallelism
   - Splits model across multiple devices while using DDP

### Environment Configuration

The training environment is automatically configured through:
- Process group initialization with NCCL backend
- Proper GPU device assignment
- Environment variable validation
- Automatic GPU count detection

## Source

This implementation is modified from the [PyTorch examples repository](https://github.com/pytorch/examples/blob/main/distributed/ddp/example.py), with additional features for containerization and cloud deployment.

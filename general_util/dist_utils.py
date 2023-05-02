import datetime
import os
import subprocess

import torch
import torch.distributed as dist
from omegaconf import DictConfig


def vanilla_torch_dist(cfg: DictConfig, backend="nccl"):
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] not in [-1, "-1"]:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])

    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=7200))
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
    cfg.device = device


def setup_slurm_distributed(cfg: DictConfig, backend="nccl", port=None):
    """
    Most code are copied from https://github.com/BIGBALLON/distribuuuu/blob/master/tutorial/mnmc_ddp_slurm.py.
    """
    num_gpus = torch.cuda.device_count()
    print(num_gpus)
    if num_gpus <= 1 or cfg.no_cuda:
        cfg.local_rank = -1
        cfg.device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = min(num_gpus, 1)
        cfg.ddp_eval = False
        return

    # Data Parallel or Model Parallel on multiple GPUs with single task.
    if int(os.environ["SLURM_NTASKS"]) == 1:
        cfg.n_gpu = num_gpus
        cfg.ddp_eval = False
        cfg.device = str(torch.device("cuda"))
        cfg.local_rank = -1
        return

    proc_id = int(os.environ["SLURM_PROCID"])
    n_tasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]

    torch.cuda.set_device(proc_id % num_gpus)

    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr

    os.environ["WORLD_SIZE"] = str(n_tasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)

    cfg.n_gpu = 1
    cfg.local_rank = int(os.environ["LOCAL_RANK"])
    # cfg.local_rank = int(os.environ["RANK"])
    cfg.world_size = int(os.environ["WORLD_SIZE"])
    cfg.device = str(torch.device("cuda", cfg.local_rank))

    dist.init_process_group(backend=backend, world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))

    # print(cfg.n_gpu, cfg.local_rank, cfg.world_size, cfg.device)
    # print(cfg.local_rank)
    cfg.local_rank = dist.get_rank()
    # print(cfg.local_rank)

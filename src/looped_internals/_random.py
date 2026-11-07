"""Random state snapshots for interrupted experiment computations."""

import random

import numpy as np
import torch


def rng_state(device=None):
    cuda_device = torch.device(device or "cuda")
    save_cuda = cuda_device.type == "cuda" and torch.cuda.is_initialized()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(cuda_device) if save_cuda else None,
    }


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device or "cuda")

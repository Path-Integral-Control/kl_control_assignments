"""GPU/CPU backend switcher.

Lets the solver and models use CuPy (GPU) or NumPy (CPU) transparently.
Call set_backend(True) before creating an MPPI instance to use the GPU.
"""

import numpy as np

USE_GPU = False


def set_backend(use_gpu: bool):
    global USE_GPU
    USE_GPU = use_gpu


def get_backend():
    if USE_GPU:
        try:
            import cupy as cp

            return cp
        except ImportError:
            print("CuPy not available, falling back to NumPy")
            return np
    return np

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

GPU_ENABLE = True

def get_xp():
    if GPU_ENABLE and CUPY_AVAILABLE:
        return cp
    return np

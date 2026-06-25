import numpy as np

try:
    import cupy as cp
    dev_count = cp.cuda.runtime.getDeviceCount()
    if dev_count and dev_count > 0:
        CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

GPU_ENABLE = True

def get_xp():
    if GPU_ENABLE and CUPY_AVAILABLE:
        return cp
    return np

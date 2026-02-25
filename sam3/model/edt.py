# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Fallback scipy Euclidean distance transform (EDT) for Mac/CPU without Triton."""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.
    Uses scipy distance_transform_edt since triton is not available on this platform.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
    """
    assert data.dim() == 3
    B, H, W = data.shape
    device = data.device
    
    # Move to CPU for SciPy
    np_data = data.detach().cpu().numpy()
    out = np.zeros_like(np_data, dtype=np.float32)
    
    # Process batch
    for i in range(B):
        # Invert mask: scipy edt expects 0 for background, 1 for foreground
        # The triton version expects distance to 0, so distance to False
        mask = (np_data[i] == 0)
        if mask.all():
            out[i] = 0.0
        elif not mask.any():
            # Large distance if no zeros
            out[i] = 1e9
        else:
            # SciPy distance_transform_edt finds distance to the nearest ZERO
            # So we pass the boolean array where True means foreground, False means background
            out[i] = distance_transform_edt(np_data[i])
            
    return torch.from_numpy(out).to(device)

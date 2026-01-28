"""
Color space conversion utilities.
"""

import numpy as np


def rgb_to_yuv(rgb):
    """
    Convert RGB to YUV color space (ITU-R BT.601).
    
    Parameters
    ----------
    rgb : numpy.ndarray
        Input RGB image with shape (H, W, 3).
        
    Returns
    -------
    tuple
        (Y, Cb, Cr) channels as separate numpy arrays.
    """
    Y = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    Cb = -0.168736 * rgb[:, :, 0] - 0.331264 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2] + 128
    Cr = 0.5 * rgb[:, :, 0] - 0.418688 * rgb[:, :, 1] - 0.081312 * rgb[:, :, 2] + 128
    return Y, Cb, Cr


def yuv_to_rgb(Y, Cb, Cr):
    """
    Convert YUV to RGB color space (ITU-R BT.601).
    
    Parameters
    ----------
    Y : numpy.ndarray
        Luminance channel.
    Cb : numpy.ndarray
        Blue-difference chroma channel.
    Cr : numpy.ndarray
        Red-difference chroma channel.
        
    Returns
    -------
    numpy.ndarray
        RGB image with shape (H, W, 3).
    """
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    rgb = np.zeros((*Y.shape, 3), dtype=np.float64)
    rgb[:, :, 0] = np.clip(R, 0, 255)
    rgb[:, :, 1] = np.clip(G, 0, 255)
    rgb[:, :, 2] = np.clip(B, 0, 255)
    return rgb

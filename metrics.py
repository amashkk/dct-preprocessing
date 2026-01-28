"""
Image quality metrics for evaluating compression results.

This module provides various metrics to assess image quality:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Edge Preservation
- Texture Preservation
- Sharpness
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim_func


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Parameters
    ----------
    img1 : numpy.ndarray
        First image (reference).
    img2 : numpy.ndarray
        Second image (distorted).
        
    Returns
    -------
    float
        PSNR value in dB.
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Parameters
    ----------
    img1 : numpy.ndarray
        First image (reference).
    img2 : numpy.ndarray
        Second image (distorted).
        
    Returns
    -------
    float
        SSIM value (0-1, higher is better).
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    if len(img1.shape) == 2:
        return ssim_func(img1, img2, data_range=255.0)
    elif len(img1.shape) == 3:
        return ssim_func(img1, img2, data_range=255.0, channel_axis=2)
    else:
        raise ValueError("Unsupported image dimensions")


def calculate_edge_preservation(original, compressed):
    """
    Calculate edge preservation ratio using Sobel edge detection.
    
    Parameters
    ----------
    original : numpy.ndarray
        Original image.
    compressed : numpy.ndarray
        Compressed/processed image.
        
    Returns
    -------
    float
        Correlation coefficient between edge maps (0-1).
    """
    if len(original.shape) == 3:
        original_gray = 0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]
        compressed_gray = 0.299 * compressed[:,:,0] + 0.587 * compressed[:,:,1] + 0.114 * compressed[:,:,2]
    else:
        original_gray = original
        compressed_gray = compressed
        
    edge_orig = np.sqrt(
        ndimage.sobel(original_gray, axis=0)**2 + 
        ndimage.sobel(original_gray, axis=1)**2
    )
    edge_comp = np.sqrt(
        ndimage.sobel(compressed_gray, axis=0)**2 + 
        ndimage.sobel(compressed_gray, axis=1)**2
    )
    
    correlation = np.corrcoef(edge_orig.flatten(), edge_comp.flatten())[0, 1]
    return correlation


def calculate_texture_preservation(original, compressed):
    """
    Calculate texture preservation using local standard deviation.
    
    Parameters
    ----------
    original : numpy.ndarray
        Original image.
    compressed : numpy.ndarray
        Compressed/processed image.
        
    Returns
    -------
    float
        Correlation coefficient between texture maps (0-1).
    """
    if len(original.shape) == 3:
        original_gray = 0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]
        compressed_gray = 0.299 * compressed[:,:,0] + 0.587 * compressed[:,:,1] + 0.114 * compressed[:,:,2]
    else:
        original_gray = original
        compressed_gray = compressed

    def local_std(img, size=5):
        mean = uniform_filter(img, size=size)
        mean_sq = uniform_filter(img**2, size=size)
        return np.sqrt(np.maximum(mean_sq - mean**2, 0))

    texture_orig = local_std(original_gray)
    texture_comp = local_std(compressed_gray)
    
    correlation = np.corrcoef(texture_orig.flatten(), texture_comp.flatten())[0, 1]
    return correlation


def calculate_sharpness(image):
    """
    Calculate image sharpness using Laplacian variance.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image.
        
    Returns
    -------
    float
        Sharpness value (higher means sharper).
    """
    if len(image.shape) == 3:
        gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    else:
        gray = image
        
    laplacian = ndimage.laplace(gray)
    sharpness = np.var(laplacian)
    return sharpness


def evaluate_all_metrics(original, compressed):
    """
    Calculate all quality metrics at once.
    
    Parameters
    ----------
    original : numpy.ndarray
        Original reference image.
    compressed : numpy.ndarray
        Compressed/processed image.
        
    Returns
    -------
    dict
        Dictionary containing all metric values.
    """
    return {
        'psnr': calculate_psnr(original, compressed),
        'ssim': calculate_ssim(original, compressed),
        'edge_preservation': calculate_edge_preservation(original, compressed),
        'texture_preservation': calculate_texture_preservation(original, compressed),
        'sharpness_original': calculate_sharpness(original),
        'sharpness_compressed': calculate_sharpness(compressed),
    }

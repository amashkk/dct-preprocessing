"""
DCT Compression implementation.

This module provides a simple DCT-based image compression/decompression
similar to JPEG, used for testing preprocessing methods.
"""

import numpy as np
from scipy.fftpack import dct, idct


class DCTCompressor:
    """
    DCT-based Image Compressor.
    
    This implements a simplified JPEG-like compression using DCT
    with configurable quality levels.
    
    Parameters
    ----------
    quality : int, default=50
        Compression quality (1-100). Lower values mean more compression
        and more artifacts.
        
    Attributes
    ----------
    block_size : int
        Size of DCT blocks (8x8).
    quantization_table : numpy.ndarray
        Quantization table derived from quality setting.
    """

    def __init__(self, quality=50):
        self.quality = max(1, min(100, quality))
        self.block_size = 8
        self.quantization_table = self._get_quantization_table(self.quality)

    def _get_quantization_table(self, quality):
        """
        Generate quantization table based on quality.
        
        Uses the standard JPEG luminance quantization table as base.
        """
        base_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)
        
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
            
        quant_table = np.floor((base_table * scale + 50) / 100)
        quant_table[quant_table < 1] = 1
        return quant_table

    def _dct2d(self, block):
        """Apply 2D DCT to a block."""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def _idct2d(self, block):
        """Apply 2D inverse DCT to a block."""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def compress_decompress(self, image):
        """
        Compress and decompress an image (simulate lossy compression).
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image (grayscale or RGB).
            
        Returns
        -------
        numpy.ndarray
            Reconstructed image after compression/decompression.
        """
        if len(image.shape) == 2:
            return self._compress_channel(image)
        elif len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for c in range(image.shape[2]):
                result[:, :, c] = self._compress_channel(image[:, :, c])
            return result
        else:
            raise ValueError("Unsupported image dimensions")

    def _compress_channel(self, channel):
        """Compress and decompress a single channel."""
        height, width = channel.shape
        
        # Pad to multiple of 8
        pad_height = (8 - height % 8) % 8
        pad_width = (8 - width % 8) % 8
        if pad_height > 0 or pad_width > 0:
            channel = np.pad(channel, ((0, pad_height), (0, pad_width)), mode='edge')
            
        padded_height, padded_width = channel.shape
        result = np.zeros_like(channel)
        
        # Process 8x8 blocks
        for y in range(0, padded_height, 8):
            for x in range(0, padded_width, 8):
                block = channel[y:y+8, x:x+8].astype(np.float64) - 128
                
                # DCT
                dct_block = self._dct2d(block)
                
                # Quantization
                quantized = np.round(dct_block / self.quantization_table)
                
                # Dequantization
                dequantized = quantized * self.quantization_table
                
                # Inverse DCT
                reconstructed = self._idct2d(dequantized)
                result[y:y+8, x:x+8] = np.clip(reconstructed + 128, 0, 255)
        
        # Remove padding
        if pad_height > 0 or pad_width > 0:
            result = result[:height, :width]
            
        return result

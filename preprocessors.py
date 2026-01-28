"""
DCT Preprocessing classes for ringing artifact reduction.

This module contains three preprocessing approaches:
1. DCTPreprocessorFixed - Original paper's fixed threshold method
2. DCTPreprocessorAdaptive - Our optimized adaptive threshold method
3. UniformPreprocessor - Global filtering baseline
"""

import numpy as np
from .utils import rgb_to_yuv, yuv_to_rgb


class DCTPreprocessorFixed:
    """
    Selective Filtering Preprocessor with Fixed Threshold.
    
    This implements the original method from Oizumi (2006) paper.
    
    Parameters
    ----------
    rho_threshold : float, default=0.3
        Threshold for the modified auto-correlation coefficient.
        Pixels with rho_mod > threshold will be filtered.
    filter_intensity : float, default=2.5
        Multiplier for filtering strength.
    window_size : int, default=9
        Size of the local window for analysis.
    delta : float, default=10
        Regularization constant to prevent division by zero.
        
    Attributes
    ----------
    filter_map : numpy.ndarray
        Map showing filter intensity applied to each pixel.
    rho_map : numpy.ndarray
        Map of modified auto-correlation coefficients.
    """

    def __init__(self, rho_threshold=0.3, filter_intensity=2.5, window_size=9, delta=10):
        self.rho_threshold = rho_threshold
        self.filter_intensity = filter_intensity
        self.window_size = window_size
        self.delta = delta
        self.filter_map = None
        self.rho_map = None

    def calculate_autocorrelation(self, data, lag):
        """Calculate autocorrelation at given lag."""
        n = len(data)
        if lag >= n or lag < 0:
            return 0
        sum_val = sum(data[i] * data[i + lag] for i in range(n - lag))
        return sum_val / (n - lag) if n > lag else 0

    def calculate_rho_mod(self, pixels):
        """
        Calculate the modified auto-correlation coefficient.
        
        rho_mod = R_xx(1) / (R_xx(0) + delta)
        
        Parameters
        ----------
        pixels : numpy.ndarray
            Local pixel values.
            
        Returns
        -------
        float
            Modified auto-correlation coefficient.
        """
        mean = np.mean(pixels)
        centered = pixels - mean
        R0 = self.calculate_autocorrelation(centered, 0)
        R1 = self.calculate_autocorrelation(centered, 1)
        if R0 + self.delta < 1e-10:
            return 0
        return R1 / (R0 + self.delta)

    def apply_stronger_lowpass_filter(self, pixels):
        """Apply a 5-tap low-pass filter."""
        if len(pixels) < 5:
            return pixels.copy()
        filtered = np.copy(pixels).astype(np.float64)
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for i in range(2, len(pixels) - 2):
            filtered[i] = np.sum(weights * pixels[i-2:i+3])
        if len(pixels) >= 3:
            weights_3 = np.array([0.25, 0.5, 0.25])
            filtered[1] = np.sum(weights_3 * pixels[0:3])
            filtered[-2] = np.sum(weights_3 * pixels[-3:])
        return filtered

    def preprocess_direction(self, image, direction='horizontal'):
        """
        Apply preprocessing in one direction.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input grayscale image.
        direction : str, default='horizontal'
            Processing direction ('horizontal' or 'vertical').
            
        Returns
        -------
        numpy.ndarray
            Preprocessed image.
        """
        if direction == 'vertical':
            image = image.T
        height, width = image.shape
        processed = np.copy(image).astype(np.float64)
        filter_map = np.zeros((height, width))
        rho_map_dir = np.zeros((height, width))

        half_window = self.window_size // 2

        for y in range(height):
            for x in range(width):
                start_x = max(0, x - half_window)
                end_x = min(width, x + half_window + 1)
                local_pixels = image[y, start_x:end_x].astype(np.float64)
                rho_mod = self.calculate_rho_mod(local_pixels)
                rho_map_dir[y, x] = rho_mod

                if rho_mod > self.rho_threshold:
                    filtered_pixels = self.apply_stronger_lowpass_filter(local_pixels)
                    center_idx = x - start_x
                    if center_idx < len(filtered_pixels):
                        original_value = image[y, x]
                        filtered_value = filtered_pixels[center_idx]
                        intensity = min(1.0, (rho_mod - self.rho_threshold) * self.filter_intensity)
                        filter_map[y, x] = intensity
                        processed[y, x] = original_value * (1 - intensity) + filtered_value * intensity

        if direction == 'vertical':
            processed = processed.T
            filter_map = filter_map.T
            rho_map_dir = rho_map_dir.T

        self.filter_map = filter_map
        if direction == 'horizontal':
            self.rho_map = rho_map_dir

        return processed

    def preprocess_image(self, image):
        """
        Preprocess the entire image.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image (grayscale or RGB).
            
        Returns
        -------
        numpy.ndarray
            Preprocessed image.
        """
        if len(image.shape) == 2:
            processed = self.preprocess_direction(image.astype(np.float64), 'horizontal')
            filter_map_h = self.filter_map.copy()
            processed = self.preprocess_direction(processed, 'vertical')
            filter_map_v = self.filter_map.copy()
            self.filter_map = np.maximum(filter_map_h, filter_map_v)
            return processed
        elif len(image.shape) == 3:
            rgb = image.astype(np.float64)
            Y, Cb, Cr = rgb_to_yuv(rgb)
            Y_processed = self.preprocess_direction(Y, 'horizontal')
            filter_map_h = self.filter_map.copy()
            Y_processed = self.preprocess_direction(Y_processed, 'vertical')
            filter_map_v = self.filter_map.copy()
            self.filter_map = np.maximum(filter_map_h, filter_map_v)
            processed = yuv_to_rgb(Y_processed, Cb, Cr)
            return processed
        else:
            raise ValueError("Unsupported image dimensions")

    def visualize_rho_map(self, image, direction='horizontal'):
        """Get the rho_mod map for visualization."""
        if self.rho_map is not None and direction == 'horizontal':
            return self.rho_map

        if len(image.shape) == 3:
            rgb = image.astype(np.float64)
            Y, _, _ = rgb_to_yuv(rgb)
            image = Y
        if direction == 'vertical':
            image = image.T

        height, width = image.shape
        rho_map = np.zeros((height, width))
        half_window = self.window_size // 2
        for y in range(height):
            for x in range(width):
                start_x = max(0, x - half_window)
                end_x = min(width, x + half_window + 1)
                local_pixels = image[y, start_x:end_x]
                rho_map[y, x] = self.calculate_rho_mod(local_pixels)
        if direction == 'vertical':
            rho_map = rho_map.T
        return rho_map


class DCTPreprocessorAdaptive:
    """
    Selective Filtering Preprocessor with Adaptive Threshold.
    
    This is our optimized version that dynamically adjusts the threshold
    based on the image's global variance.
    
    Parameters
    ----------
    base_rho_threshold : float, default=0.3
        Base threshold value.
    adaptive_threshold : bool, default=True
        Whether to use adaptive threshold.
    adaptive_range_variance : tuple, default=(500, 3000)
        Variance range for threshold interpolation.
    adaptive_range_threshold : tuple, default=(0.2, 0.5)
        Threshold range for interpolation.
    filter_intensity : float, default=2.5
        Multiplier for filtering strength.
    window_size : int, default=9
        Size of the local window for analysis.
    delta : float, default=10
        Regularization constant.
        
    Attributes
    ----------
    current_rho_threshold : float
        The dynamically calculated threshold for current image.
    filter_map : numpy.ndarray
        Map showing filter intensity applied to each pixel.
    rho_map : numpy.ndarray
        Map of modified auto-correlation coefficients.
    """

    def __init__(self, base_rho_threshold=0.3, adaptive_threshold=True,
                 adaptive_range_variance=(500, 3000),
                 adaptive_range_threshold=(0.2, 0.5),
                 filter_intensity=2.5, window_size=9, delta=10):

        self.filter_intensity = filter_intensity
        self.window_size = window_size
        self.delta = delta
        self.filter_map = None
        self.rho_map = None

        self.base_rho_threshold = base_rho_threshold
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_range_variance = adaptive_range_variance
        self.adaptive_range_threshold = adaptive_range_threshold
        self.current_rho_threshold = self.base_rho_threshold

    def _calculate_global_variance(self, image_y):
        """Calculate the global variance of Y channel."""
        return np.var(image_y)

    def calculate_autocorrelation(self, data, lag):
        """Calculate autocorrelation at given lag."""
        n = len(data)
        if lag >= n or lag < 0:
            return 0
        sum_val = sum(data[i] * data[i + lag] for i in range(n - lag))
        return sum_val / (n - lag) if n > lag else 0

    def calculate_rho_mod(self, pixels):
        """Calculate the modified auto-correlation coefficient."""
        mean = np.mean(pixels)
        centered = pixels - mean
        R0 = self.calculate_autocorrelation(centered, 0)
        R1 = self.calculate_autocorrelation(centered, 1)
        if R0 + self.delta < 1e-10:
            return 0
        return R1 / (R0 + self.delta)

    def apply_stronger_lowpass_filter(self, pixels):
        """Apply a 5-tap low-pass filter."""
        if len(pixels) < 5:
            return pixels.copy()
        filtered = np.copy(pixels).astype(np.float64)
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for i in range(2, len(pixels) - 2):
            filtered[i] = np.sum(weights * pixels[i-2:i+3])
        if len(pixels) >= 3:
            weights_3 = np.array([0.25, 0.5, 0.25])
            filtered[1] = np.sum(weights_3 * pixels[0:3])
            filtered[-2] = np.sum(weights_3 * pixels[-3:])
        return filtered

    def preprocess_direction(self, image, direction='horizontal'):
        """Apply preprocessing in one direction with adaptive threshold."""
        if direction == 'vertical':
            image = image.T
        height, width = image.shape
        processed = np.copy(image).astype(np.float64)
        filter_map = np.zeros((height, width))
        rho_map_dir = np.zeros((height, width))

        half_window = self.window_size // 2

        for y in range(height):
            for x in range(width):
                start_x = max(0, x - half_window)
                end_x = min(width, x + half_window + 1)
                local_pixels = image[y, start_x:end_x].astype(np.float64)
                rho_mod = self.calculate_rho_mod(local_pixels)
                rho_map_dir[y, x] = rho_mod

                # Use dynamic threshold
                if rho_mod > self.current_rho_threshold:
                    filtered_pixels = self.apply_stronger_lowpass_filter(local_pixels)
                    center_idx = x - start_x
                    if center_idx < len(filtered_pixels):
                        original_value = image[y, x]
                        filtered_value = filtered_pixels[center_idx]
                        intensity = min(1.0, (rho_mod - self.current_rho_threshold) * self.filter_intensity)
                        filter_map[y, x] = intensity
                        processed[y, x] = original_value * (1 - intensity) + filtered_value * intensity

        if direction == 'vertical':
            processed = processed.T
            filter_map = filter_map.T
            rho_map_dir = rho_map_dir.T

        self.filter_map = filter_map
        if direction == 'horizontal':
            self.rho_map = rho_map_dir

        return processed

    def preprocess_image(self, image, verbose=False):
        """
        Preprocess the entire image with adaptive threshold.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image (grayscale or RGB).
        verbose : bool, default=False
            Whether to print progress messages.
            
        Returns
        -------
        numpy.ndarray
            Preprocessed image.
        """
        # Extract Y channel
        if len(image.shape) == 2:
            Y = image.astype(np.float64)
        elif len(image.shape) == 3:
            rgb = image.astype(np.float64)
            Y, Cb, Cr = rgb_to_yuv(rgb)
        else:
            raise ValueError("Unsupported image dimensions")

        # Adaptive threshold calculation
        if self.adaptive_threshold:
            variance = self._calculate_global_variance(Y)
            v_min, v_max = self.adaptive_range_variance
            t_min, t_max = self.adaptive_range_threshold
            self.current_rho_threshold = np.interp(variance, [v_min, v_max], [t_min, t_max])
            
            if verbose:
                print(f"  → Y channel global variance: {variance:.2f}")
                print(f"  → Dynamic threshold: {self.current_rho_threshold:.4f}")
        else:
            self.current_rho_threshold = self.base_rho_threshold

        # Apply filtering
        if len(image.shape) == 2:
            processed = self.preprocess_direction(Y, 'horizontal')
            filter_map_h = self.filter_map.copy()
            processed = self.preprocess_direction(processed, 'vertical')
            filter_map_v = self.filter_map.copy()
            self.filter_map = np.maximum(filter_map_h, filter_map_v)
            return processed
        elif len(image.shape) == 3:
            Y_processed = self.preprocess_direction(Y, 'horizontal')
            filter_map_h = self.filter_map.copy()
            Y_processed = self.preprocess_direction(Y_processed, 'vertical')
            filter_map_v = self.filter_map.copy()
            self.filter_map = np.maximum(filter_map_h, filter_map_v)
            processed = yuv_to_rgb(Y_processed, Cb, Cr)
            return processed

    def visualize_rho_map(self, image, direction='horizontal'):
        """Get the rho_mod map for visualization."""
        if self.rho_map is not None and direction == 'horizontal':
            return self.rho_map

        if len(image.shape) == 3:
            rgb = image.astype(np.float64)
            Y, _, _ = rgb_to_yuv(rgb)
            image = Y
        if direction == 'vertical':
            image = image.T

        height, width = image.shape
        rho_map = np.zeros((height, width))
        half_window = self.window_size // 2
        for y in range(height):
            for x in range(width):
                start_x = max(0, x - half_window)
                end_x = min(width, x + half_window + 1)
                local_pixels = image[y, start_x:end_x]
                rho_map[y, x] = self.calculate_rho_mod(local_pixels)
        if direction == 'vertical':
            rho_map = rho_map.T
        return rho_map


class UniformPreprocessor:
    """
    Uniform (Global) Filtering Preprocessor.
    
    This applies the same low-pass filter to the entire image,
    used as a baseline for comparison.
    
    Parameters
    ----------
    filter_strength : float, default=0.5
        Blending factor between original and filtered image (0-1).
    """

    def __init__(self, filter_strength=0.5):
        self.filter_strength = filter_strength

    def apply_uniform_filter(self, pixels):
        """Apply a 5-tap low-pass filter."""
        if len(pixels) < 5:
            return pixels.copy()
        filtered = np.copy(pixels).astype(np.float64)
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for i in range(2, len(pixels) - 2):
            filtered[i] = np.sum(weights * pixels[i-2:i+3])
        if len(pixels) >= 3:
            weights_3 = np.array([0.25, 0.5, 0.25])
            filtered[1] = np.sum(weights_3 * pixels[0:3])
            filtered[-2] = np.sum(weights_3 * pixels[-3:])
        return filtered

    def preprocess_direction(self, image, direction='horizontal'):
        """Apply uniform filtering in one direction."""
        if direction == 'vertical':
            image = image.T
        height, width = image.shape
        processed = np.copy(image).astype(np.float64)
        for y in range(height):
            row = image[y, :].astype(np.float64)
            filtered_row = self.apply_uniform_filter(row)
            processed[y, :] = (1 - self.filter_strength) * row + self.filter_strength * filtered_row
        if direction == 'vertical':
            processed = processed.T
        return processed

    def preprocess_image(self, image):
        """
        Preprocess the entire image with uniform filtering.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image (grayscale or RGB).
            
        Returns
        -------
        numpy.ndarray
            Preprocessed image.
        """
        if len(image.shape) == 2:
            processed = self.preprocess_direction(image.astype(np.float64), 'horizontal')
            processed = self.preprocess_direction(processed, 'vertical')
            return processed
        elif len(image.shape) == 3:
            rgb = image.astype(np.float64)
            Y, Cb, Cr = rgb_to_yuv(rgb)
            Y_processed = self.preprocess_direction(Y, 'horizontal')
            Y_processed = self.preprocess_direction(Y_processed, 'vertical')
            processed = yuv_to_rgb(Y_processed, Cb, Cr)
            return processed
        else:
            raise ValueError("Unsupported image dimensions")

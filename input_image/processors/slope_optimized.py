"""
DEM processing functions with CPU optimization using Numba.
"""

import os
import numpy as np
import rasterio
import numba
from numba import prange
from pathlib import Path

from ..config import UNIVERSAL_DTYPE, UNIVERSAL_NODATA, GEOTIFF_EXT, UNIVERSAL_CRS


@numba.njit(parallel=True)
def numba_convolve(array, kernel):
    """
    Fast convolution implementation using Numba.
    
    Args:
        array: Input 2D array
        kernel: Convolution kernel (3x3)
        
    Returns:
        Convolved array
    """
    height, width = array.shape
    k_height, k_width = kernel.shape
    pad_height, pad_width = k_height // 2, k_width // 2
    
    # Create output array
    result = np.zeros((height, width), dtype=np.float32)
    
    # Perform convolution
    for i in prange(pad_height, height - pad_height):
        for j in range(pad_width, width - pad_width):
            # Skip computation for NaN values
            if np.isnan(array[i, j]):
                result[i, j] = np.nan
                continue
                
            # Apply kernel
            sum_val = 0.0
            for ki in range(k_height):
                for kj in range(k_width):
                    ii = i + ki - pad_height
                    jj = j + kj - pad_width
                    if not np.isnan(array[ii, jj]):
                        sum_val += array[ii, jj] * kernel[ki, kj]
            
            result[i, j] = sum_val
    
    # Handle edge pixels
    for i in range(height):
        for j in range(width):
            if i < pad_height or i >= height - pad_height or j < pad_width or j >= width - pad_width:
                result[i, j] = np.nan
    
    return result


@numba.njit
def calculate_slope_numba(dem_array, cell_size, neighbors=8, units="degrees"):
    """
    Calculate slope from a DEM array using Horn's method with Numba optimization.
    
    Parameters:
    -----------
    dem_array : numpy.ndarray
        Input DEM array
    cell_size : tuple
        (x, y) cell size in map units
    neighbors : int, optional
        Number of neighboring cells (4 or 8)
    units : str, optional
        Units for slope calculation ('degrees' or 'grade')
        
    Returns:
    --------
    numpy.ndarray
        Slope array
    """
    dx, dy = cell_size
    
    if neighbors == 4:
        # 4-connected neighbors
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32) / (8.0 * dx)
        kernel_y = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=np.float32) / (8.0 * dy)
    else:
        # 8-connected neighbors (Horn's method)
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32) / (8.0 * dx)
        kernel_y = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=np.float32) / (8.0 * dy)
    
    # Calculate gradients using fast convolution
    dzdx = numba_convolve(dem_array, kernel_x)
    dzdy = numba_convolve(dem_array, kernel_y)
    
    # Calculate slope
    slope = np.sqrt(dzdx**2 + dzdy**2)
    
    # Convert to degrees if requested
    if units == "degrees":
        # Use arctan for each element
        for i in prange(slope.shape[0]):
            for j in range(slope.shape[1]):
                if not np.isnan(slope[i, j]):
                    slope[i, j] = np.degrees(np.arctan(slope[i, j]))
    
    return slope


def compute_slope(dem_path, bands_dir):
    """Compute slope from DEM using Numba-optimized Horn's method
    
    Args:
        dem_path (str or Path): Path to the DEM file
        bands_dir (str or Path): Directory for saving output
        
    Returns:
        Path: Path to the generated slope file
    """
    print("\n=== Starting compute_slope (optimized) ===")
    dem_path = Path(dem_path)
    bands_dir = Path(bands_dir)
    
    print(f"Loading DEM from: {dem_path}")
    
    # Get the DEM metadata and read the data
    with rasterio.open(dem_path) as src:
        dem_meta = src.meta.copy()
        transform = src.transform
        dem_array = src.read(1).astype(np.float32)  # Ensure float32 type for Numba
        
        # Calculate cell sizes
        cell_size_x = abs(transform[0])
        cell_size_y = abs(transform[4])
        
        print(f"DEM cell sizes - X: {cell_size_x}, Y: {cell_size_y}")
        print(f"DEM loaded, shape: {dem_array.shape}")
        
        # Check if cell sizes are significantly different
        cell_ratio = cell_size_x / cell_size_y
        if abs(cell_ratio - 1.0) > 0.01:  # If more than 1% difference
            print(f"Warning: Non-square pixels detected (X/Y ratio = {cell_ratio:.4f})")
    
    # Compute slope using optimized Horn's method
    print("Computing slope using Numba-optimized Horn's method...")
    cell_size = (cell_size_x, cell_size_y)
    
    # Use a warm-up run to trigger JIT compilation
    print("Warming up Numba JIT compiler...")
    small_array = dem_array[0:100, 0:100] if dem_array.shape[0] > 100 and dem_array.shape[1] > 100 else dem_array
    _ = calculate_slope_numba(small_array, cell_size, neighbors=8, units="degrees")
    
    # Now run the full calculation
    print("Running full slope calculation...")
    slope_array = calculate_slope_numba(dem_array, cell_size, neighbors=8, units="degrees")
    
    print("Slope computation complete")
    
    # Define output path
    slope_path = bands_dir / f"slope{GEOTIFF_EXT}"
    print(f"Writing slope to: {slope_path}")
    
    # Update metadata for slope raster
    dem_meta.update({
        'dtype': UNIVERSAL_DTYPE,
        'count': 1,
        'nodata': UNIVERSAL_NODATA
    })
    
    # Write the slope raster
    with rasterio.open(slope_path, 'w', **dem_meta) as dst:
        dst.write(slope_array[np.newaxis, :, :].astype(dem_meta['dtype']))
        dst.set_band_description(1, "Terrain slope (degrees)")
    
    # Print statistics
    valid_slope = slope_array[~np.isnan(slope_array)]
    if len(valid_slope) > 0:
        print(f"Slope statistics - min: {np.min(valid_slope):.2f}°, max: {np.max(valid_slope):.2f}°, mean: {np.mean(valid_slope):.2f}°")
    else:
        print("Warning: No valid slope values found")
    
    print("=== Completed compute_slope (optimized) ===\n")
    
    return slope_path 
"""
EDTW processing functions with optimization using Numba.
This is an optimized version of the distance transform calculation.
"""

import os
import numpy as np
import rasterio
from geopy.distance import geodesic
import numba
from numba import prange
from ..config import UNIVERSAL_DTYPE, UNIVERSAL_NODATA
from ..utils.print_info import log_export_info

def get_cell_size(water_path):
    """Get pixel size in meters based on geographic coordinates"""
    with rasterio.open(water_path) as src:
        transform = src.transform
        crs = src.crs
        water = src.read(1)

        # Get pixel width and height in degrees
        pixel_width_deg = transform.a
        pixel_height_deg = -transform.e

        # Center of the raster to estimate latitude
        center_lat = src.bounds.top - (src.height // 2) * pixel_height_deg
        center_lon = src.bounds.left + (src.width // 2) * pixel_width_deg

        # Approximate meters per pixel using geodesic distance
        pixel_width_m = geodesic(
            (center_lat, center_lon),
            (center_lat, center_lon + pixel_width_deg)
        ).meters

        pixel_height_m = geodesic(
            (center_lat, center_lon),
            (center_lat + pixel_height_deg, center_lon)
        ).meters

    print(f"Pixel size: {pixel_width_m:.2f} m x {pixel_height_m:.2f} m")
    return pixel_width_m, pixel_height_m

@numba.njit(parallel=True)
def optimized_edt(binary_mask, sampling=(1.0, 1.0)):
    """
    Optimized Euclidean distance transform using Numba.
    
    Args:
        binary_mask: Binary mask where 0 represents feature pixels
        sampling: Pixel size in each dimension for proper scaling
    
    Returns:
        Distance transform array
    """
    height, width = binary_mask.shape
    output = np.full(binary_mask.shape, np.inf, dtype=np.float32)
    
    # First pass - initialize points where binary_mask == 0
    for i in prange(height):
        for j in range(width):
            if binary_mask[i, j] == 0:
                output[i, j] = 0.0
    
    max_dist = height * sampling[0] + width * sampling[1]
    
    # Process all pixels
    for i in prange(height):
        for j in range(width):
            if output[i, j] > 0:
                # Search for nearest feature pixel
                min_dist = max_dist
                
                # Use a windowed search to improve performance
                search_radius = int(min(50, min(height, width) // 2))  # Limit search radius
                
                i_start = max(0, i - search_radius)
                i_end = min(height, i + search_radius + 1)
                j_start = max(0, j - search_radius)
                j_end = min(width, j + search_radius + 1)
                
                for ni in range(i_start, i_end):
                    for nj in range(j_start, j_end):
                        if binary_mask[ni, nj] == 0:
                            # Calculate squared distance with proper scaling
                            dy = (i - ni) * sampling[0]
                            dx = (j - nj) * sampling[1]
                            dist = np.sqrt(dy*dy + dx*dx)
                            if dist < min_dist:
                                min_dist = dist
                
                output[i, j] = min_dist
    
    return output

def compute_euclidean_distance(water_path, output_dir):
    """
    Compute Euclidean distance to nearest non-zero pixel using
    optimized implementation with Numba.
    
    Args:
        water_path (str): Path to binary water raster
        output_dir (str): Directory for saving results
        
    Returns:
        numpy.ndarray: Distance transform
    """
    print("\n=== Starting compute_euclidean_distance (optimized) ===")
    
    # Read binary water raster
    with rasterio.open(water_path) as src:
        binary_raster = src.read(1)
        meta = src.meta
    
    # Get cell size in meters
    pixel_width_m, pixel_height_m = get_cell_size(water_path)

    water_pixels = np.sum(binary_raster == 1)
    total_pixels = binary_raster.size
    print(f"Computing distance from {water_pixels} water pixels ({water_pixels/total_pixels*100:.2f}% of raster)")
    
    # Check if water pixels are too sparse - if so, use the standard distance transform
    water_percentage = water_pixels / total_pixels
    
    if water_percentage < 0.01 or binary_raster.size > 25000000:  # If < 1% water or large raster
        print("Using scipy's distance_transform_edt due to sparse water or large raster...")
        from scipy.ndimage import distance_transform_edt
        distance = distance_transform_edt(binary_raster == 0, sampling=(pixel_width_m, pixel_height_m))
    else:
        print("Using Numba-optimized distance transform...")
        # Compute optimized distance transform
        distance = optimized_edt(binary_raster, sampling=(pixel_height_m, pixel_width_m))
    
    # Save distance transform to intermediate directory if provided
    edtw_path = os.path.join(output_dir, 'edtw.tif')
    print(f"Writing EDTW to: {edtw_path}")
    
    meta.update({
        'dtype': UNIVERSAL_DTYPE,
        'nodata': UNIVERSAL_NODATA
    })
    
    with rasterio.open(edtw_path, 'w', **meta) as dst:
        dst.write(distance[np.newaxis, :, :])
        
    # Print information about the resulting GeoTIFF
    with rasterio.open(edtw_path) as src:
        log_export_info("EDTW", src, is_ee=False)
    
    print(f"Distance min: {distance.min()}, max: {distance.max()}, mean: {distance.mean():.2f}")
    print("=== Completed compute_euclidean_distance (optimized) ===\n")
    return distance 
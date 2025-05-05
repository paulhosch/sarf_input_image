"""
EDTW processing functions.
"""

import os
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from pysheds.grid import Grid
from ..config import UNIVERSAL_DTYPE, UNIVERSAL_NODATA
from ..utils.print_info import log_export_info
from geopy.distance import geodesic
def get_cell_size(water_path):
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
def compute_euclidean_distance(water_path, output_dir):
    """Compute Euclidean distance to nearest non-zero pixel
    
    Args:
        water_path (str): Path to binary water raster
        output_dir (str): Directory for saving results
        
    Returns:
        numpy.ndarray: Distance transform
    """
    print("\n=== Starting compute_euclidean_distance ===")
    
    # Read binary water raster
    with rasterio.open(water_path) as src:
        binary_raster = src.read(1)
        meta = src.meta
    
    pixel_width_m, pixel_height_m = get_cell_size(water_path)

    water_pixels = np.sum(binary_raster == 1)
    total_pixels = binary_raster.size
    print(f"Computing distance from {water_pixels} water pixels ({water_pixels/total_pixels*100:.2f}% of raster)")
    
    # Compute distance (in pixels)
    distance = distance_transform_edt(binary_raster == 0, sampling=(pixel_width_m, pixel_height_m))
    
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
    print("=== Completed compute_euclidean_distance ===\n")
    return distance

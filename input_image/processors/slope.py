"""
DEM processing functions.
"""

import os
import numpy as np
import rasterio
import richdem as rd
from pathlib import Path

from ..config import UNIVERSAL_DTYPE, UNIVERSAL_NODATA, GEOTIFF_EXT, UNIVERSAL_CRS


def compute_slope(dem_path, bands_dir):
    """Compute slope from DEM using richdem
    
    Args:
        dem_path (str or Path): Path to the DEM file
        bands_dir (str or Path): Directory for saving output
        
    Returns:
        Path: Path to the generated slope file
    """
    print("\n=== Starting compute_slope ===")
    dem_path = Path(dem_path)
    bands_dir = Path(bands_dir)
    
    print(f"Loading DEM from: {dem_path}")
    
    # Get the DEM metadata first to check for different X and Y resolutions
    with rasterio.open(dem_path) as src:
        dem_meta = src.meta.copy()
        transform = src.transform
        
        # Calculate cell sizes
        cell_size_x = abs(transform[0])
        cell_size_y = abs(transform[4])
        
        print(f"DEM cell sizes - X: {cell_size_x}, Y: {cell_size_y}")
        
        # Check if cell sizes are significantly different
        cell_ratio = cell_size_x / cell_size_y
        if abs(cell_ratio - 1.0) > 0.01:  # If more than 1% difference
            print(f"Warning: Non-square pixels detected (X/Y ratio = {cell_ratio:.4f})")
    
    # Load the DEM using richdem
    dem_rd = rd.LoadGDAL(str(dem_path), no_data=UNIVERSAL_NODATA)
    print(f"DEM loaded, shape: {dem_rd.shape}")
    
    # Set the correct cell sizes in richdem for accurate slope calculation
    if hasattr(dem_rd, 'geotransform'):
        # Save original geotransform
        orig_transform = dem_rd.geotransform
        
        # Apply correct cell sizes
        dem_rd.geotransform = (orig_transform[0], cell_size_x, orig_transform[2], 
                              orig_transform[3], -cell_size_y, orig_transform[5])
    
    # Compute slope using richdem (in degrees)
    print("Computing slope using richdem...")
    slope_rd = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
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
        # Convert richdem array to numpy and reshape to 3D for rasterio
        slope_np = np.array(slope_rd, dtype=dem_meta['dtype'])
        dst.write(slope_np[np.newaxis, :, :])
        dst.set_band_description(1, "Terrain slope (degrees)")
    
    # Print statistics
    valid_slope = slope_np[~np.isnan(slope_np)]
    if len(valid_slope) > 0:
        print(f"Slope statistics - min: {np.min(valid_slope):.2f}°, max: {np.max(valid_slope):.2f}°, mean: {np.mean(valid_slope):.2f}°")
    else:
        print("Warning: No valid slope values found")
    
    print("=== Completed compute_slope ===\n")
    
    return slope_path
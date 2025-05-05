"""
Label processing functions.
"""

import os
import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
from shapely.geometry import mapping
from ..config import UNIVERSAL_DTYPE, UNIVERSAL_CRS, UNIVERSAL_NODATA, GEOTIFF_EXT
from ..utils.print_info import log_export_info

def rasterize_label(ground_truth_dir, output_dit):
    """Create local label image from ground truth shapefile
    
    Args:
        ground_truth_dir (str): Directory containing ground truth shapefile
        output_dit (str): Directory to save output files
        
    Returns:
        str: Path to the created label image
    """
    print("\n=== Starting get_label ===")
    print(f"Ground truth directory: {ground_truth_dir}")
    print(f"Output directory: {output_dit}")
    
    # Output path for label image
    label_path = os.path.join(output_dit, f"label{GEOTIFF_EXT}")
    
    # Check if ground truth directory exists
    if not os.path.exists(ground_truth_dir):
        print(f"Ground truth delineation directory {ground_truth_dir} does not exist. Skipping label creation.")
        print("=== Completed get_label (skipped) ===\n")
        return None
    
    # Check if there are shapefiles in the ground truth directory
    shp_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.shp')]
    if not shp_files:
        print("No ground truth shapefile found. Skipping label creation.")
        print("=== Completed get_label (skipped) ===\n")
        return None
    
    # Read ground truth shapefile
    gt_path = os.path.join(ground_truth_dir, shp_files[0])
    print(f"Reading ground truth shapefile: {gt_path}")
    gt = gpd.read_file(gt_path)
    print(f"Ground truth CRS: {gt.crs}")
    print(f"Ground truth geometry type: {gt.geometry.iloc[0].geom_type}")
    
    # Read VV_pre.tif to get correct projection and resolution
    vv_pre_path = os.path.join(output_dit, f"VV_pre{GEOTIFF_EXT}")
    if not os.path.exists(vv_pre_path):
        print(f"VV_pre{GEOTIFF_EXT} not found at {vv_pre_path}. Unable to create label image.")
        print("=== Completed get_label (failed) ===\n")
        return None
    
    print(f"Reading reference image: {vv_pre_path}")
    with rasterio.open(vv_pre_path) as src:
        # Get metadata from VV_pre
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        shape = src.shape
        
        # Create empty raster with same dimensions as VV_pre
        # Using float32 for consistency with universal settings
        label_array = np.zeros(shape, dtype=UNIVERSAL_DTYPE)
        
        # Reproject ground truth to match VV_pre CRS if needed
        if gt.crs != crs:
            print(f"Reprojecting ground truth from {gt.crs} to {crs}")
            gt = gt.to_crs(crs)
        
        # Rasterize ground truth polygon (1 for flooded, 0 for not flooded)
        print("Rasterizing ground truth polygon...")
        shapes = [(geom, 1.0) for geom in gt.geometry]  # Using 1.0 for float32
        label_array = features.rasterize(
            shapes=shapes,
            out=label_array,
            transform=transform,
            fill=0.0,  # Using 0.0 for float32
            all_touched=False,
            dtype=UNIVERSAL_DTYPE
        )
        
        # Update metadata for label image using universal settings
        meta.update({
            'dtype': UNIVERSAL_DTYPE,
            'count': 1,
            'nodata': UNIVERSAL_NODATA,
            'crs': UNIVERSAL_CRS
        })
        
        # Write label image
        print(f"Writing label image to: {label_path}")
        with rasterio.open(label_path, 'w', **meta) as dst:
            dst.write(label_array, 1)
            dst.set_band_description(1, "Ground truth flood extent (1=flooded)")
        
        # Print information about the resulting GeoTIFF
        with rasterio.open(label_path) as label_src:
            log_export_info("Label", label_src, is_ee=False)
    
    print(f"Label image created at: {label_path}")
    print("=== Completed get_label ===\n")
    
    return label_path


"""
Earth Engine export functions.
"""

import os
import glob
import math
import shutil
from pathlib import Path

import ee
import geemap
import rasterio
from rasterio.merge import merge

from ..config import UNIVERSAL_DTYPE, UNIVERSAL_CRS, UNIVERSAL_NODATA
from ..utils.print_info import log_export_info


def export_large_ee_image(ee_image, output_path, aoi_ee, scale=10):
    """Export large Earth Engine image to local GeoTIFF using parallel downloading
    
    This function overcomes the 50MB download limit by splitting the area into smaller
    tiles, downloading them in parallel, and then merging them back together.
    
    Args:
        ee_image (ee.Image): Image to export (single band)
        output_path (str): Path to save output GeoTIFF
        aoi_ee (ee.Geometry): Area of interest
        scale (int, optional): Export resolution in meters. If None, uses the image's native scale.
    """
    # Get native scale if none provided
    if scale is None:
        scale = ee_image.projection().nominalScale().getInfo()
        print(f"Exporting in native ee.Image scale of {scale} meters")
    
    print(f"\nExporting large image using parallel downloading to: {output_path}")
    
    # Print information about Earth Engine image
    log_export_info(os.path.basename(output_path), ee_image, is_ee=True)
    
    # Make sure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary directory for tiles
    temp_dir = os.path.join(output_dir, "temp_tiles")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Get the bounds of the AOI to calculate fishnet parameters
        bounds = aoi_ee.bounds().getInfo()['coordinates'][0]
        coords = bounds
        west = min(p[0] for p in coords)
        south = min(p[1] for p in coords)
        east = max(p[0] for p in coords)
        north = max(p[1] for p in coords)
        
        # Calculate width and height in degrees
        width_deg = east - west
        height_deg = north - south
        
        # Calculate grid size to get at least 10 tiles
        # We'll aim for a roughly square grid
        grid_size = math.ceil(math.sqrt(10))
        
        # Calculate interval sizes
        h_interval = width_deg / grid_size
        v_interval = height_deg / grid_size
        
        # Use 10% overlap between tiles
        delta = min(h_interval, v_interval) * 0.1
        
        # Create a fishnet grid over the AOI
        fishnet = geemap.fishnet(aoi_ee, h_interval=h_interval, v_interval=v_interval, delta=delta)
        
        tile_count = fishnet.size().getInfo()
        print(f"Created fishnet with {tile_count} tiles")
        
        # Project the image to ensure it has a fixed projection
        projected_image = ee_image.reproject(crs=UNIVERSAL_CRS, scale=scale)
        
        # Download tiles in parallel
        print("Downloading image tiles in parallel...")
        geemap.download_ee_image_tiles_parallel(
            projected_image, 
            fishnet, 
            out_dir=temp_dir, 
            #scale=scale,
            crs=UNIVERSAL_CRS
        )
        
        # List all downloaded tiles
        tiles = glob.glob(os.path.join(temp_dir, "*.tif"))
        if not tiles:
            raise Exception("No tiles were downloaded")
        
        print(f"Downloaded {len(tiles)} tiles, merging them...")
        
        # Open all tiles with rasterio
        src_files_to_mosaic = []
        for tile in tiles:
            src = rasterio.open(tile)
            src_files_to_mosaic.append(src)
        
        # Merge tiles
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        # Copy the metadata from the first file
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        # Update metadata with universal settings
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": UNIVERSAL_CRS,
            "dtype": UNIVERSAL_DTYPE,
            "nodata": UNIVERSAL_NODATA
        })
        
        # Convert data to proper dtype if needed
        mosaic = mosaic.astype(UNIVERSAL_DTYPE)
        
        # Write merged result
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Close all source files
        for src in src_files_to_mosaic:
            src.close()
            
        # Print information about the resulting GeoTIFF
        with rasterio.open(output_path) as src:
            log_export_info(os.path.basename(output_path), src, is_ee=False)
            
        print(f"Successfully merged tiles and saved to {output_path}")
        
    except Exception as e:
        print(f"Error during parallel download: {e}")
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary files in {temp_dir}")
    
    return output_path
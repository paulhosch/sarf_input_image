"""
DEM processing functions.
"""

import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import mapping
from rasterio.io import MemoryFile
from ..config import UNIVERSAL_DTYPE, UNIVERSAL_CRS, UNIVERSAL_NODATA, GEOTIFF_EXT
from ..utils.print_info import log_export_info
from ..utils import logger, log_execution
import requests
import zipfile
import io
from pathlib import Path
import geopandas as gpd
import math
from tqdm import tqdm

@log_execution
def process_dem(dem_tiles_dir, aoi_gdf, output_dir):
    """Process DEM tiles: merge, clip to AOI, convert units, and match to SAR imagery
    
    Args:
        dem_tiles_dir (str): Directory containing DEM tiles
        aoi_gdf (GeoDataFrame): Area of interest
        output_dir (str, optional): Directory for saving output files
        
    Returns:
        str: Path to the processed DEM file
    """
    logger.info("=== Starting process_dem ===")
    
    # Find all TIF files (using the extension from config)
    dem_files = [os.path.join(dem_tiles_dir, f) for f in os.listdir(dem_tiles_dir) 
                if f.endswith(GEOTIFF_EXT)]
    logger.info(f"Found {len(dem_files)} DEM tiles in {dem_tiles_dir}")
    
    if not dem_files:
        logger.error(f"Error: No DEM tiles found in {dem_tiles_dir}")
        raise FileNotFoundError(f"No DEM tiles found in {dem_tiles_dir}")
    
    
    # Read DEM tiles
    logger.info("Opening DEM tile files...")
    dem_src_files = [rasterio.open(f) for f in dem_files]
    
    # Log information about the first DEM tile
    log_export_info("DEM Tile (First)", dem_src_files[0], is_ee=False)
    
    # Merge DEM tiles
    logger.info("Merging DEM tiles...")
    dem_mosaic, transform = merge(dem_src_files)
    logger.info(f"Merged DEM shape: {dem_mosaic.shape}")
    
    # Convert elevation from cm to m using the factor from config
    unit_conversion_factor = 100.0
    logger.info(f"Converting elevation units using factor 1/{unit_conversion_factor}...")
    dem_mosaic = dem_mosaic / unit_conversion_factor
    
    # Get metadata from first file
    meta = dem_src_files[0].meta.copy()
    meta.update({
        'height': dem_mosaic.shape[1],
        'width': dem_mosaic.shape[2],
        'transform': transform,
    })
    
    # Handle nodata values 
    original_nodata = dem_src_files[0].nodata
    if original_nodata is not None:
        logger.info(f"Converting original nodata value {original_nodata}")
        # Scale the original nodata if needed
        scaled_nodata = original_nodata / unit_conversion_factor
        # Replace with same nodata value
        dem_mosaic = np.where(dem_mosaic == scaled_nodata, original_nodata, dem_mosaic)
    
    
    logger.info(f"Final DEM shape: {dem_mosaic.shape}")
    logger.info(f"DEM min: {dem_mosaic.min():.2f}m, max: {dem_mosaic.max():.2f}m, mean: {dem_mosaic.mean():.2f}m")
    logger.info(f"DEM CRS: {meta['crs']}")
    logger.info(f"DEM transform: {meta['transform']}")
    
    # Close source files
    for src in dem_src_files:
        src.close()
    logger.info("Source files closed")
        
    # Saving DEM 
    dem_path = os.path.join(output_dir, f"dem{GEOTIFF_EXT}")
    logger.info(f"Writing processed DEM to: {dem_path}")
    with rasterio.open(dem_path, 'w', **meta) as dst:
        dst.write(dem_mosaic)
        dst.set_band_description(1, "Digital Elevation Model (m)")
    
    # Print information about the output DEM
    with rasterio.open(dem_path) as src:
        log_export_info("DEM (processed)", src, is_ee=False)

    logger.info("=== Completed process_dem ===\n")
    
    return dem_path

def get_FathomDEM_tiles(aoi_gdf, dem_dir, token=None):
    """
    Download FathomDEM tiles from Zenodo that cover the given AOI.
    
    Parameters:
    -----------
    aoi_gdf : GeoDataFrame
        GeoDataFrame containing the Area of Interest
    dem_dir : Path or str
        Directory where to save the downloaded DEM tiles
    token : str, optional
        Zenodo API token. If None, will look for a token in the environment
        or use a default token.
    
    Returns:
    --------
    list
        List of paths to the downloaded tiles
    """
    # Ensure dem_dir is a Path
    dem_dir = Path(dem_dir)
    dem_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure AOI is in lat/lon
    aoi_gdf = aoi_gdf.to_crs(epsg=4326)
    
    # Get the bounding box
    min_lon, min_lat, max_lon, max_lat = aoi_gdf.total_bounds
    
    # Determine integer tile indices
    lat_start = math.floor(min_lat)
    lat_end = math.floor(max_lat)
    lon_start = math.floor(min_lon)
    lon_end = math.floor(max_lon)
    
    # Generate list of required tiles
    tile_names = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            lat_prefix = 'n' if lat >= 0 else 's'
            lon_prefix = 'e' if lon >= 0 else 'w'
            tile_filename = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}.tif"
            tile_names.append(tile_filename)
    
    logger.info(f"Required tiles: {', '.join(tile_names)}")
    
    # Assign tiles to zip archives (FathomDEM organizes tiles in 30x30° zip archives)
    zip_mapping = {}
    for tile in tile_names:
        # Parse lat/lon from tile name
        lat_sign = 1 if tile[0] == 'n' else -1
        lon_sign = 1 if tile[3] == 'e' else -1
        lat = lat_sign * int(tile[1:3])
        lon = lon_sign * int(tile[4:7])
        
        # Compute SW corner of containing zip (multiples of 30)
        sw_lat = (math.floor(lat / 30) * 30)
        sw_lon = (math.floor(lon / 30) * 30)
        ne_lat = sw_lat + 30
        ne_lon = sw_lon + 30
        
        # Format zip name
        sw = f"{('n' if sw_lat>=0 else 's')}{abs(sw_lat):02d}{('e' if sw_lon>=0 else 'w')}{abs(sw_lon):03d}"
        ne = f"{('n' if ne_lat>=0 else 's')}{abs(ne_lat):02d}{('e' if ne_lon>=0 else 'w')}{abs(ne_lon):03d}"
        zip_name = f"{sw}-{ne}_FathomDEM_v1-0.zip"
        
        if zip_name not in zip_mapping:
            zip_mapping[zip_name] = []
        zip_mapping[zip_name].append(tile)
    
    # Use default token if none provided
    if token is None:
        token = "nLOGIrtBhyrJEo4kECFwHs7I4vzWRnNQIznfIIya0iLjecZxP6m3HeVDYNYz"
    
    # Fetch file list from Zenodo
    record_id = 14511570  # FathomDEM record ID on Zenodo
    headers = {'Authorization': f'Bearer {token}'}
    url = f'https://zenodo.org/api/records/{record_id}'
    
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        record_files = r.json()['files']
    except Exception as e:
        logger.error(f"Error fetching record files: {e}")
        return []
    
    # Download and extract required tiles
    downloaded_tiles = []
    
    for zip_name, tiles in zip_mapping.items():
        # Find zip URL
        file_info = next((f for f in record_files if f['key'] == zip_name), None)
        if not file_info:
            logger.warning(f"Warning: {zip_name} not found in record.")
            continue
        
        logger.info(f"Downloading {zip_name}...")
        try:
            # Stream the content with progress bar
            resp = requests.get(file_info['links']['self'], headers=headers, stream=True)
            resp.raise_for_status()
            
            # Get total file size from headers
            total_size = int(resp.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            # Create a buffer to store the downloaded content
            buffer = io.BytesIO()
            
            # Download with progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                      desc=f"{zip_name}", bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
                for chunk in resp.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive chunks
                        buffer.write(chunk)
                        pbar.update(len(chunk))
            
            # Reset buffer pointer to start
            buffer.seek(0)
            
            # Open the zip file from the buffer
            z = zipfile.ZipFile(buffer)
            
            # Extract the needed tiles with progress
            logger.info(f"Extracting tiles from {zip_name}...")
            for tile in tqdm(tiles, desc="Extracting tiles", bar_format='{l_bar}{bar:30}{r_bar}'):
                tile_path = dem_dir / tile
                # Skip if already downloaded
                if tile_path.exists():
                    logger.info(f"Tile {tile} already exists, skipping.")
                    downloaded_tiles.append(tile_path)
                    continue
                
                if tile in z.namelist():
                    with z.open(tile) as src, open(tile_path, 'wb') as dst:
                        dst.write(src.read())
                    downloaded_tiles.append(tile_path)
                else:
                    logger.warning(f"Tile {tile} not found in {zip_name}.")
            
            z.close()
            buffer.close()
        except Exception as e:
            logger.error(f"Error downloading/extracting {zip_name}: {e}")
    
    return downloaded_tiles


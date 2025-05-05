"""
HAND processing functions.
"""

import numpy as np
import rasterio
from pysheds.grid import Grid
from pathlib import Path
from ..config import UNIVERSAL_DTYPE, UNIVERSAL_NODATA


def stream_burn(dem_path, water_path, bands_dir, burn_depth=20):
    """Stream burn the DEM by lowering elevation at water pixels
    
    Args:
        dem_path (str or Path): Path to the DEM file
        water_path (str or Path): Path to the binary water raster file
        bands_dir (str or Path): Directory for saving output files
        burn_depth (int, optional): Depth to burn streams in meters. Defaults to 20.
        
    Returns:
        Path: Path to the burned DEM file
    """
    print("\n=== Starting stream_burn ===")
    dem_path = Path(dem_path)
    water_path = Path(water_path)
    bands_dir = Path(bands_dir)
    
    print(f"Loading DEM from: {dem_path}")
    print(f"Loading water raster from: {water_path}")
    
    # Load DEM and water raster
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        dem_meta = src.meta.copy()
    
    with rasterio.open(water_path) as src:
        water_raster = src.read(1)
    
    # Make a copy of the DEM to modify
    burned_dem = dem.copy()
    
    # Burn streams by lowering elevation at water pixels 
    burned_dem[water_raster == 1] -= burn_depth
    
    # Save burned DEM
    burned_dem_path = bands_dir / 'burned_dem.tif'
    print(f"Writing burned DEM to: {burned_dem_path}")
    
    # Update metadata for burned DEM
    dem_meta.update({
        'dtype': UNIVERSAL_DTYPE,
        'nodata': UNIVERSAL_NODATA
    })
    
    with rasterio.open(burned_dem_path, 'w', **dem_meta) as dst:
        dst.write(burned_dem[np.newaxis, :, :])
        dst.set_band_description(1, "Stream-burned DEM (m)")
        
    # Print statistics
    if np.any(~np.isnan(burned_dem)):
        print(f"Burned DEM stats - min: {np.nanmin(burned_dem):.2f}m, max: {np.nanmax(burned_dem):.2f}m")
        print(f"Burned {np.sum(water_raster == 1)} water pixels by {burn_depth}m")
    else:
        print("Warning: No valid data in burned DEM")
        
    print("=== Completed stream_burn ===\n")
    return burned_dem_path

def compute_hand(dem_path, water_path, burned_dem_path, bands_dir):
    """Compute Height Above Nearest Drainage (HAND) using pysheds
    
    Args:
        dem_path (str or Path): Path to the DEM file
        bands_dir (str or Path): Directory for saving output files
        
    Returns:
        Path: Path to the HAND file
    """
    print("\n=== Starting compute_hand ===")
    
    # Check if required files exist
    if not water_path.exists():
        raise FileNotFoundError(f"Water raster not found at {water_path}")
    if not burned_dem_path.exists():
        raise FileNotFoundError(f"Burned DEM not found at {burned_dem_path}")
    
    print(f"Using files: \n  DEM: {dem_path}\n  Water: {water_path}\n  Burned DEM: {burned_dem_path}")
    
    # Initialize grid
    grid = Grid.from_raster(str(dem_path))
    
    # Read raster data
    dem_data = grid.read_raster(str(dem_path))
    burned_dem_data = grid.read_raster(str(burned_dem_path))
    osm_water = grid.read_raster(str(water_path))
    osm_water = grid.view(osm_water, nodata_out=0)

    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(burned_dem_data)

    # Fill depressions in DEM
    print("Filling depressions in DEM...")
    flooded_dem = grid.fill_depressions(pit_filled_dem)
        
    # Resolve flats in DEM
    print("Resolving flats in DEM...")
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Fill depressions in DEM
    print("Filling depressions in DEM...")
    inflated_dem = grid.fill_depressions(inflated_dem)
        
    # Resolve flats in DEM
    print("Resolving flats in DEM...")
    inflated_dem = grid.resolve_flats(inflated_dem)

    # Compute flow direction
    print("Computing flow direction...")
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, flats=-1, pits=-2, nodata_out=0)
    
    # Compute flow accumulation
    print("Computing flow accumulation...")
    flow_accumulation = grid.accumulation(fdir)

    # Compute HAND
    print("Computing HAND values...")
    hand = grid.compute_hand(fdir, dem_data, osm_water > 0)
    hand_array = grid.view(hand)
    
    # Replace infinities and NaNs with a valid nodata value
    hand_array = np.where(np.isfinite(hand_array), hand_array, UNIVERSAL_NODATA)
    
    # Get metadata from DEM for output file
    with rasterio.open(dem_path) as src:
        dem_meta = src.meta.copy()
    
    # Save HAND 
    hand_path = bands_dir / 'hand.tif'
    print(f"Writing HAND to: {hand_path}")

    # Update metadata for HAND raster
    dem_meta.update({
        'dtype': UNIVERSAL_DTYPE,
        'nodata': UNIVERSAL_NODATA
    })
    
    with rasterio.open(hand_path, 'w', **dem_meta) as dst:
        dst.write(hand_array[np.newaxis, :, :])
        dst.set_band_description(1, "Height Above Nearest Drainage (m)")
    
    # Get statistics
    valid_hand = hand_array[np.isfinite(hand_array)]
    if len(valid_hand) > 0:
        min_val = np.min(valid_hand)
        max_val = np.max(valid_hand)
        mean_val = np.mean(valid_hand)
        print(f"HAND stats - min: {min_val:.2f}m, max: {max_val:.2f}m, mean: {mean_val:.2f}m")
        print(f"Valid pixels: {len(valid_hand)} of {hand_array.size} ({len(valid_hand)/hand_array.size:.1%})")
    else:
        print("WARNING: No valid HAND values found")
    
    print("=== Completed compute_hand ===\n")
    
    return hand_path 
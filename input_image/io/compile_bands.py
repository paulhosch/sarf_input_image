import os
import rasterio
import numpy as np
import geopandas as gpd
from pathlib import Path
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from ..config import GEOTIFF_EXT

def compile_input_image(bands_dir, output_path, aoi=None, reference_band=None, resampling_method=Resampling.bilinear):
    """Compile all rasters in bands_dir into a single multi-band input image
    
    Args:
        bands_dir: Directory containing band files
        output_path: Path to save the compiled input image
        aoi: Either a path to a shapefile or a GeoDataFrame for clipping the output (optional)
        reference_band: Band file to use as reference (if None, uses the first band)
        resampling_method: Resampling method to use (default: bilinear)
        
    Returns:
        dict: Dictionary with band information
    """
    bands_dir = Path(bands_dir)
    output_path = Path(output_path)
    
    # Get all GeoTIFF files in the bands directory
    band_files = sorted([f for f in bands_dir.glob(f"*{GEOTIFF_EXT}")])
    
    if not band_files:
        return None
        
    # Determine reference band
    reference_file = None
    if isinstance(reference_band, str):
        reference_candidates = [f for f in band_files if f.stem.lower() == reference_band.lower()]
        if reference_candidates:
            reference_file = reference_candidates[0]
            print(f"Using specified reference band: {reference_file.name}")
    
    if reference_file is None:
        reference_file = band_files[0]
        print(f"Using first band as reference: {reference_file.name}")
    
    # Get reference metadata
    with rasterio.open(reference_file) as src:
        reference_profile = src.profile.copy()
        reference_crs = src.crs
        reference_transform = src.transform
        reference_width = src.width
        reference_height = src.height
    
    # Update profile for output
    reference_profile.update({
        'count': len(band_files),
        'dtype': 'float32'
    })
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Band info dictionary
    band_info = {}
    band_descriptions = []
    
    # Create the output file and process bands
    with rasterio.open(output_path, 'w', **reference_profile) as dst:
        for i, file_path in enumerate(band_files):
            band_name = file_path.stem
            band_name_lower = band_name.lower()
            
            with rasterio.open(file_path) as src:
                # Check if this band needs reprojection or resampling
                needs_alignment = (src.crs != reference_crs or 
                                  src.width != reference_width or 
                                  src.height != reference_height or
                                  src.transform != reference_transform)
                
                if needs_alignment:
                    print(f"Aligning band: {band_name}")
                    
                    # Create destination array
                    dst_band = np.zeros((reference_height, reference_width), dtype='float32')
                    
                    # Read source data
                    src_band = src.read(1)
                    
                    # Reproject/resample band to match reference
                    reproject(
                        source=src_band,
                        destination=dst_band,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=reference_transform,
                        dst_crs=reference_crs,
                        resampling=resampling_method
                    )
                    
                    band_data = dst_band
                else:
                    # No reprojection/resampling needed
                    band_data = src.read(1).astype(np.float32)
                
                # Write band
                dst.write(band_data, i+1)
                
                # Set band description
                description = band_name.upper()
                band_descriptions.append(description)
                dst.set_band_description(i+1, description)
                
                # Add band info
                band_info[band_name_lower] = {
                    'description': description,
                    'path': str(file_path),
                    'band_idx': i,
                    'category': 'Other'
                }
            
            # Free memory
            del band_data
    
    # Clip to AOI if provided
    if aoi is not None:
        # Determine if aoi is a path or GeoDataFrame
        if isinstance(aoi, (str, Path)):
            print(f"Reading AOI from shapefile: {aoi}")
            aoi_gdf = gpd.read_file(aoi)
        else:
            # Assume it's already a GeoDataFrame
            print("Using provided GeoDataFrame for AOI")
            aoi_gdf = aoi
        
        # Check if shapefile has valid geometries
        if aoi_gdf.empty:
            print("Warning: AOI has no geometries, skipping clip")
            return band_info
            
        # Reproject GeoDataFrame to match the raster CRS if needed
        if aoi_gdf.crs != reference_crs:
            print(f"Reprojecting AOI from {aoi_gdf.crs} to {reference_crs}")
            aoi_gdf = aoi_gdf.to_crs(reference_crs)
        
        # Create a temporary file for the clipped output
        temp_output = output_path.with_suffix('.temp' + output_path.suffix)
        
        # Open the compiled image
        with rasterio.open(output_path) as src:
            # Get the geometries in the correct format for rasterio
            shapes = [geom for geom in aoi_gdf.geometry]
            
            # Perform the mask operation
            out_image, out_transform = mask(src, shapes, crop=True)
            
            # Update the metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Write the clipped data to the temporary file
            with rasterio.open(temp_output, "w", **out_meta) as dest:
                dest.write(out_image)
                
                # Copy all band descriptions
                for i in range(1, len(band_descriptions) + 1):
                    desc = src.descriptions[i-1] if i-1 < len(src.descriptions) else None
                    if desc:
                        dest.set_band_description(i, desc)
        
        # Replace the original file with the clipped one
        import shutil
        temp_output.replace(output_path)
        print(f"Successfully clipped to AOI")
    
    return band_info 
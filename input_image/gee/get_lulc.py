"""
Land cover processing functions.
"""

import ee
import os
from .export import export_large_ee_image


def get_esri_lulc(aoi_ee, output_dir, year='2023'):
    """Get land cover data and export to GeoTIFF using parallel download
    
    Args:
        aoi_ee (ee.Geometry): Area of interest
        output_dir (str):  output directory
        site_id (str): Site identifier
        year (str): Year for land cover data, default '2023'
        
    Returns:
        str: Path to the exported land cover file
    """
    print(f"\n=== Starting get_landcover  ===")
    
    # Define output path
    output_path = os.path.join(output_dir, "land_cover.tif")
    
    # Access ESRI land cover collection
    print(f"Accessing ESRI Global Land Cover collection for {year}...")
    esri_lulc = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
    
    # Get land cover for the specified year
    print(f"Filtering for {year} land cover data...")
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    lulc = esri_lulc.filterDate(start_date, end_date).mosaic()
    
    # Clip to AOI
    print("Clipping land cover to AOI...")
    lulc_clipped = lulc.clip(aoi_ee).rename('land_cover')
    
    # Export using parallel download method
    print(f"Exporting land cover to {output_path}...")
    export_large_ee_image(lulc_clipped, output_path, aoi_ee, scale=10)
    
    print(f"Land cover exported to {output_path}")
    print(f"=== Completed get_landcover ===\n")
    
    return output_path 
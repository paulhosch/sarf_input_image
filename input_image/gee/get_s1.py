"""
Sentinel-1 data processing functions.
"""

import ee
from shapely.geometry import mapping
import os
from ..gee.export import export_large_ee_image

def get_VV_VH(aoi_ee, pre_date, post_date, output_dir):
    """Get Sentinel-1 VV and VH bands for pre and post event dates and export as individual TIFs
    
    Args:
        aoi_ee (ee.Geometry): Area of interest
        pre_date (str): Pre-event date (YYYY-MM-DD)
        post_date (str): Post-event date (YYYY-MM-DD)
        output_dir (str): Directory to save  files
        
    Returns:
        ee.Geometry: Earth Engine geometry of AOI
    """
    print("\n=== Starting get_VV_VH ===")
    print(f"Pre-event date: {pre_date}")
    print(f"Post-event date: {post_date}")
    
    # Sentinel-1 collection
    print("Accessing Sentinel-1 collection...")
    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')
    
    # Get pre-event image
    print(f"Filtering for pre-event image ({pre_date})...")
    pre_image = s1_collection \
        .filterDate(pre_date, (ee.Date(pre_date).advance(1, 'day'))) \
        .filterBounds(aoi_ee) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .select(['VV', 'VH'])\
        .mosaic()    
    # Get post-event image
    print(f"Filtering for post-event image ({post_date})...")
    post_image = s1_collection \
        .filterDate(post_date, (ee.Date(post_date).advance(1, 'day'))) \
        .filterBounds(aoi_ee) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .select(['VV', 'VH'])\
        .mosaic()
    # Clip to AOI
    print("Clipping images to AOI...")
    pre_image = pre_image.clip(aoi_ee)
    post_image = post_image.clip(aoi_ee)
    
    # Export individual bands as separate TIF files

    
    print("Exporting individual polarization bands...")
    # Export VV pre
    vv_pre = pre_image.select('VV').rename('VV_pre')
    export_large_ee_image(vv_pre, os.path.join(output_dir, 'VV_pre.tif'), aoi_ee, scale=10)
    
    # Export VH pre
    vh_pre = pre_image.select('VH').rename('VH_pre')
    export_large_ee_image(vh_pre, os.path.join(output_dir, 'VH_pre.tif'), aoi_ee, scale=10)
    
    # Export VV post
    vv_post = post_image.select('VV').rename('VV_post')
    export_large_ee_image(vv_post, os.path.join(output_dir, 'VV_post.tif'), aoi_ee, scale=10)
    
    # Export VH post
    vh_post = post_image.select('VH').rename('VH_post')
    export_large_ee_image(vh_post, os.path.join(output_dir, 'VH_post.tif'), aoi_ee, scale=10)
    
    print("=== Completed get_VV_VH ===\n")
    
    return output_dir

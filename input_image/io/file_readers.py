import os
import json
import geopandas as gpd
from datetime import datetime
import ee
from shapely.geometry import mapping

def read_aoi(aoi_dir):
    """Read the AOI shapefile
    
    Args:
        aoi_dir (str): Directory containing the AOI shapefile
        
    Returns:
        GeoDataFrame: AOI geometry
    
    Raises:
        FileNotFoundError: If no shapefile is found
    """
    print("\n=== Starting read_aoi ===")
    print(f"Looking for shapefiles in: {aoi_dir}")
    
    shp_files = [f for f in os.listdir(aoi_dir) if f.endswith('.shp')]
    
    if not shp_files:
        print(f"Error: No shapefile found in {aoi_dir}")
        raise FileNotFoundError(f"No shapefile found in {aoi_dir}")
    
    aoi_path = os.path.join(aoi_dir, shp_files[0])
    print(f"Reading AOI shapefile: {aoi_path}")
    
    aoi_gdf = gpd.read_file(aoi_path)
    print(f"AOI CRS: {aoi_gdf.crs}")
    print(f"AOI geometry type: {aoi_gdf.geometry.iloc[0].geom_type}")
    print(f"AOI bounds: {aoi_gdf.total_bounds}")

    # Convert AOI to ee.Geometry
    print("Converting AOI to Earth Engine geometry...")
    aoi_geom = aoi_gdf.geometry.iloc[0]
    aoi_coords = list(mapping(aoi_geom)['coordinates'][0])
    aoi_ee = ee.Geometry.Polygon(aoi_coords)
    
    print("=== Completed read_aoi ===\n")
    
    return aoi_gdf, aoi_ee


def read_dates(dates_dir):
    """Read pre and post event dates from JSON file
    
    Args:
        dates_dir (str): Path to the dates directory
        
    Returns:
        tuple: (pre_event, post_event) dates
    """
    print("\n=== Starting read_dates ===")
    
    dates_file = os.path.join(dates_dir, 'dates.json')
    
    print(f"Reading dates from: {dates_file}")
    
    with open(dates_file, 'r') as f:
        dates = json.load(f)
    
    pre_event = dates.get('pre_event')
    post_event = dates.get('post_event')
    
    print(f"Pre-event date: {pre_event}")
    print(f"Post-event date: {post_event}")
    
    time_diff = None
    try:
        # Try to calculate date difference if dates are in YYYY-MM-DD format
        from datetime import datetime
        pre_date = datetime.strptime(pre_event, '%Y-%m-%d')
        post_date = datetime.strptime(post_event, '%Y-%m-%d')
        time_diff = (post_date - pre_date).days
        print(f"Time between dates: {time_diff} days")
    except:
        print("Could not calculate time difference (invalid date format)")
    
    print("=== Completed read_dates ===\n")
    
    return pre_event, post_event 
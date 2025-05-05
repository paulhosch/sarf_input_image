"""
OpenStreetMap water features processing functions.
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
import osmnx as ox
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import box
from ..config import UNIVERSAL_CRS
from pathlib import Path

def get_osm_water(dem_path, sea_water_path=None):
    """Get OSM water features within DEM extent. 
    This is important to enable the computation of HAND for pixels inside the AOI, 
    tha drain to a channel outside the AOI

    Args:
        dem_path (str or Path): Path to the DEM file
        sea_water_path (str or Path, optional): Path to pre-processed sea water shapefile
        
    Returns:
        GeoDataFrame: OSM water features including sea water if provided
    """
    print("\n=== Starting get_osm_water ===")
    
    # Get bounds from DEM
    with rasterio.open(dem_path) as src:
        bounds = src.bounds  # left, bottom, right, top
    
    # Create polygon from bounds
    bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    polygon = box(*bbox)
    print(f"DEM bbox: {bbox}")
    
    # Define water-related tags to query
    water_tags = {
        'natural': ['water'],
        'waterway': ['river', 'stream', 'canal', 'drain', 'tidal_channel'],
        'landuse': ['reservoir', 'basin', 'coastline']
    }
    print(f"Using water tags: {water_tags}")
    
    all_features = []
    
    # Query each tag separately
    for category, tags in water_tags.items():
        if isinstance(tags, list):
            for tag in tags:
                tag_dict = {category: tag}
                try:
                    print(f"Querying OSM for {tag_dict}...")
                    gdf = ox.features_from_polygon(polygon, tag_dict)
                    print(f"Retrieved {len(gdf)} features for {tag_dict}")
                    all_features.append(gdf)
                except Exception as e:
                    print(f"Warning: Failed for tags {tag_dict}: {e}")
        else:
            # Handle boolean tags (like waterway=True)
            tag_dict = {category: tags}
            try:
                print(f"Querying OSM for {tag_dict}...")
                gdf = ox.features_from_polygon(polygon, tag_dict)
                print(f"Retrieved {len(gdf)} features for {tag_dict}")
                all_features.append(gdf)
            except Exception as e:
                print(f"Warning: Failed for tags {tag_dict}: {e}")
    
    # Load pre-processed sea water shapefile if provided
    if sea_water_path is not None:
        try:
            print(f"Loading sea water polygons from: {sea_water_path}")
            sea_water_gdf = gpd.read_file(sea_water_path)
            
            # Clip sea water to the DEM extent
            print("Clipping sea water polygons to DEM extent")
            dem_bbox_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=UNIVERSAL_CRS)
            sea_water_gdf = sea_water_gdf.to_crs(UNIVERSAL_CRS)
            sea_water_clipped = gpd.clip(sea_water_gdf, dem_bbox_gdf)
            
            print(f"Added {len(sea_water_clipped)} sea water polygons")
            
            # Add the sea water features to the list
            if not sea_water_clipped.empty:
                all_features.append(sea_water_clipped)
        except Exception as e:
            print(f"Warning: Failed to load sea water from {sea_water_path}: {e}")
    
    # Combine all features or create empty GeoDataFrame if none found
    if all_features:
        water_gdf = gpd.GeoDataFrame(pd.concat(all_features, ignore_index=True), crs=UNIVERSAL_CRS)
        print(f"Combined {len(water_gdf)} total water features")
    else:
        water_gdf = gpd.GeoDataFrame(geometry=[], crs=UNIVERSAL_CRS)
        print("Warning: No OSM water features found in DEM extent")
    
    # Convert to same CRS 
    water_gdf = water_gdf.to_crs(UNIVERSAL_CRS)
    
    # Filter to valid geometry types
    valid_types = ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']
    water_gdf = water_gdf[water_gdf.geometry.geom_type.isin(valid_types)]
    print(f"After filtering, {len(water_gdf)} valid water features remain")
    print("=== Completed get_osm_water ===\n")
    
    return water_gdf


def rasterize_osm_water(osm_water, dem_path, bands_dir):
    """Rasterize OSM water features to binary raster
    
    Args:
        osm_water (GeoDataFrame): OSM water features
        dem_path (str or Path): Path to the DEM file to get metadata from
        bands_dir (str or Path): Directory for saving intermediate results
        
    Returns:
        numpy.ndarray: Binary water raster
    """
    print("\n=== Starting rasterize_osm_water ===")
    print(f"Rasterizing {len(osm_water)} water features")
    
    # Load metadata from DEM file
    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()
        height = meta['height']
        width = meta['width']
        transform = meta['transform']
    
    print(f"Loaded metadata from DEM: {dem_path}")
    print(f"Raster dimensions: {height} x {width} pixels")
    
    # Rasterize OSM water features
    if not osm_water.empty:
        shapes = [(geom, 1) for geom in osm_water.geometry]
        water_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        print(f"Rasterized {len(shapes)} shapes")
        water_pixels = np.sum(water_raster == 1)
        print(f"Water pixels: {water_pixels} ({water_pixels/(height*width)*100:.2f}% of raster)")
    else:
        # Create empty raster if no water features
        water_raster = np.zeros((height, width), dtype=np.uint8)
        print("Created empty water raster (no features)")
    
    # Save raster to bands directory
    bands_dir = Path(bands_dir)
    osm_water_path = bands_dir / 'osm_water.tif'
    print(f"Writing OSM water raster to: {osm_water_path}")
    
    with rasterio.open(osm_water_path, 'w', **{
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': np.uint8,
        'crs': meta['crs'],
        'transform': transform,
        'nodata': 0
    }) as dst:
        dst.write(water_raster[np.newaxis, :, :])        

    print("=== Completed rasterize_osm_water ===\n")
    return osm_water_path 
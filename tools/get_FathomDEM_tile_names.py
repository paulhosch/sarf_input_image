# %% Imports
import geopandas as gpd
import numpy as np

# %% Define function to get FathomDEM tile names
def get_fathomdem_tile_names(aoi_path):
    # FathomDEM uses 1x1 degree tiles, so we need to round latitudes and longitudes to integers
    aoi_gdf = gpd.read_file(aoi_path)
    aoi_gdf.plot()

    # Get bounding box (minx, miny, maxx, maxy) of the AOI
    minx, miny, maxx, maxy = aoi_gdf.total_bounds

    print(f"Raw Bounding Box: {minx}, {miny}, {maxx}, {maxy}")

    # Round down to get tile indices
    min_lat = int(np.floor(miny))
    max_lat = int(np.floor(maxy))
    min_lon = int(np.floor(minx))
    max_lon = int(np.floor(maxx))

    print(f"\nTile coverage:")
    print(f"Latitude range: {min_lat}° to {max_lat}°")
    print(f"Longitude range: {min_lon}° to {max_lon}°")

    # Generate a list of all required tiles
    required_tiles = []
    for lat in range(min_lat, max_lat + 1):
        for lon in range(min_lon, max_lon + 1):
            # FathomDEM naming convention: n45e002.tif
            # 'n' for north latitude, 's' for south latitude
            # 'e' for east longitude, 'w' for west longitude
            lat_prefix = 'n' if lat >= 0 else 's'
            lon_prefix = 'e' if lon >= 0 else 'w'
            
            # Use absolute values and ensure proper formatting
            lat_str = f"{lat_prefix}{abs(lat)}"
            
            # Format longitude with leading zeros (3 digits)
            lon_str = f"{lon_prefix}{abs(lon):03d}"
            
            tile_filename = f"{lat_str}{lon_str}.tif"
            required_tiles.append(tile_filename)

    print(f"\nTotal number of required tiles: {len(required_tiles)}")
    print("\nRequired FathomDEM tiles:")
    for tile in required_tiles:
        print(tile)

# %% Run the function
aoi_path = '/Users/paulhosch/Library/CloudStorage/OneDrive-Persönlich/Research/flood_mapping_with_RF_and_SAR/data/aoi/valencia /EMSR773_AOI01_DEL_PRODUCT_areaOfInterestA_v1_cropped.shp'
get_fathomdem_tile_names(aoi_path)

# %%

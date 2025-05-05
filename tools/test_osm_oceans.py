# %% Imports
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import contextily as ctx
import pandas as pd

# %% Define area of interest
# You can replace these coordinates with your area of interest
min_lon, min_lat = -0.5, 50.5  # Southwest England coast
max_lon, max_lat = 1.5, 51.5

# Create bounding box
bbox = (min_lon, min_lat, max_lon, max_lat)
polygon = box(*bbox)

print(f"Area of interest: {bbox}")

# %% Get coastline features
print("Fetching coastline features...")
coastline_tags = {'natural': 'coastline'}

try:
    coastlines = ox.features_from_polygon(polygon, coastline_tags)
    print(f"Retrieved {len(coastlines)} coastline features")
except Exception as e:
    print(f"Error fetching coastlines: {e}")
    coastlines = gpd.GeoDataFrame()

# %% Get ocean features
print("Fetching ocean features...")
ocean_tags = [
    {'natural': 'bay'},
    {'natural': 'strait'},
    {'place': 'sea'},
    {'natural': 'water', 'water': 'ocean'}
]

all_ocean_features = []

for tags in ocean_tags:
    try:
        print(f"Querying OSM for {tags}...")
        features = ox.features_from_polygon(polygon, tags)
        print(f"Retrieved {len(features)} features for {tags}")
        all_ocean_features.append(features)
    except Exception as e:
        print(f"Warning: Failed for tags {tags}: {e}")

# Combine all features or create empty GeoDataFrame if none found
if all_ocean_features:
    oceans = gpd.GeoDataFrame(pd.concat(all_ocean_features, ignore_index=True))
    print(f"Combined {len(oceans)} total ocean features")
else:
    oceans = gpd.GeoDataFrame()
    print("No ocean features found")

# %% Create a simple map visualization
fig, ax = plt.figure(figsize=(12, 10), dpi=300), plt.gca()

# Plot land area (white)
land = gpd.GeoDataFrame(geometry=[polygon], crs=coastlines.crs if not coastlines.empty else "EPSG:4326")
land.plot(ax=ax, color='white', edgecolor='black')

# Plot oceans (blue)
if not oceans.empty:
    oceans.plot(ax=ax, color='lightblue', alpha=0.7)
    print("Plotted ocean features")

# Plot coastlines
if not coastlines.empty:
    coastlines.plot(ax=ax, color='blue', linewidth=1)
    print("Plotted coastline features")

# Add basemap for context
try:
    ctx.add_basemap(ax, crs=coastlines.crs.to_string() if not coastlines.empty else "EPSG:4326", 
                    source=ctx.providers.OpenStreetMap.Mapnik)
    print("Added basemap")
except Exception as e:
    print(f"Could not add basemap: {e}")

# Set title and labels
plt.title("OSM Coastlines and Ocean Features")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# %% Save the map
output_path = "osm_oceans_map.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Map saved to {output_path}")

plt.show()

# %% Display feature count summary
print("\nFeature counts:")
if not coastlines.empty:
    print(f"Coastlines: {len(coastlines)}")
else:
    print("Coastlines: 0")

if not oceans.empty:
    print(f"Ocean features: {len(oceans)}")
    
    # Count by type
    if 'natural' in oceans.columns:
        print("\nOcean features by natural type:")
        print(oceans['natural'].value_counts())
    
    if 'place' in oceans.columns:
        print("\nOcean features by place type:")
        print(oceans['place'].value_counts())
else:
    print("Ocean features: 0")

# %% 
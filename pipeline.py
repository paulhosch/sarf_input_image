# %% Imports
import warnings 
# warnings.filterwarnings('ignore')
from pathlib import Path

# input_image modules
from input_image.gee.initialize import initialize_ee
from input_image.gee.get_s1 import get_VV_VH
from input_image.gee.get_lulc import get_esri_lulc

from input_image.io.file_readers import read_aoi, read_dates
from input_image.io.compile_bands import compile_input_image

from input_image.processors.sar import compute_sar_derivatives
from input_image.processors.label import rasterize_label
from input_image.processors.dem import process_dem, get_FathomDEM_tiles
from input_image.processors.slope import compute_slope
from input_image.processors.water import get_osm_water, rasterize_osm_water
from input_image.processors.edtw import compute_euclidean_distance
from input_image.processors.hand import compute_hand, stream_burn

from input_image.vis.plot_band import plot_bands
from input_image.vis.plot_image_stack import plot_image_stack
from input_image.vis.plut_multi_band import plot_multi_bands

# Initialize Earth Engine
print("\n=== Earth Engine Setup ===")
# Try to get project ID from environment variable
project_id = os.getenv('EARTHENGINE_PROJECT_ID')

# Initialize Earth Engine
if not initialize_ee(project_id):
    print("\nError: Earth Engine initialization failed. Please follow these steps:")
    print("1. Run 'earthengine authenticate' in your terminal")
    print("2. Set up a Google Cloud project and enable Earth Engine API")
    print("3. Set your project ID using:")
    print("   export EARTHENGINE_PROJECT_ID='your-project-id'")
    print("\nFor more details, visit: https://developers.google.com/earth-engine/guides/auth")
    sys.exit(1)

# %% Set Up Case Study
# ------------------------------------------------------------
# (1) Define the Case Study Name
site_id = 'valencia'      

# ------------------------------------------------------------
# (2) Set up your  data directory and populate with case study data
# when done, the directory should look like this (site_id and x,y,z are arbitrary):
""" 
data/
--- if AOI is near the coast, download global seawater polygons from
--- https://osmdata.openstreetmap.de/data/water-polygons.html and place in:
    global_seawater/
        water_polygons.shp
        water_polygons.dbf
        water_polygons.shx
        ...
└── case_studies/
    └──  {site_id}/
        ├── aoi/
            └── xx.shp
        ├── dem_tiles/          (can also be automatically populated, see get_FathomDEM_tiles())
            ├── y1.tif
            └── y2.tif
        ├── ground_truth/
            └── zz.shp
        ├── dates.json          (see example_dates.json)
        └── input_image/        (will be automatically created and populated)
"""
# ------------------------------------------------------------
# (3) Point to your data directory
data_dir = '../../data'
# ------------------------------------------------------------
# (4) Optional: Path to pre-processed sea water shapefile (coastline-derived)
# Set to None to use only OSM water features
sea_water_path = Path(data_dir) / 'global_seawater/water_polygons.shp'  
# ------------------------------------------------------------

# Define directory structure 
case_study_dir = Path(data_dir) / 'case_studies' / site_id
aoi_dir = case_study_dir / 'aoi'
dem_dir = case_study_dir / 'dem_tiles'
ground_truth_dir = case_study_dir / 'ground_truth'

input_image_dir = case_study_dir / 'input_image' #here will the final input image be stored
bands_dir = input_image_dir / 'bands' #during processing we store the bands as individual tif files here
plot_dir = input_image_dir / 'plots'

# Create directory if it doesn't exist
Path(input_image_dir).mkdir(parents=True, exist_ok=True)
Path(bands_dir).mkdir(parents=True, exist_ok=True)
Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(dem_dir).mkdir(parents=True, exist_ok=True)

# Read AOI
aoi_gdf, aoi_ee = read_aoi(aoi_dir)

# Read Dates
pre_event_date, post_event_date = read_dates(case_study_dir)

 #%% Download DEM tiles from FathomDEM if needed (very timeconsuming consider manually poulating the dem_dir)
#dem_tiles = get_FathomDEM_tiles(aoi_gdf, dem_dir)

# %% Get Sentinel-1 VV and VH bands
get_VV_VH(aoi_ee, pre_event_date, post_event_date, bands_dir)

# %% Compute SAR derivatives
compute_sar_derivatives(bands_dir)

# %% Create label image
rasterize_label(ground_truth_dir, bands_dir)

# %% Create land cover image
get_esri_lulc(aoi_ee, bands_dir, year='2023')

#%% Mosaic, clip, and convert DEM
dem_path = process_dem(dem_dir, aoi_gdf, bands_dir)


#%% Compute slope
compute_slope(dem_path, bands_dir)

#%% Create OSM water image
water_gdf = get_osm_water(dem_path, sea_water_path)
water_path = rasterize_osm_water(water_gdf, dem_path, bands_dir)

#%% Compute Euclidean distance to water
compute_euclidean_distance(water_path, bands_dir)

#%% Stream burn the DEM
burned_dem_path = stream_burn(dem_path, water_path, bands_dir, burn_depth=20)

#%% Compute HAND
compute_hand(dem_path, water_path, burned_dem_path, bands_dir)

# %% Compile all bands into a single input image
compile_input_image(bands_dir, input_image_dir / 'input_image.tif', aoi_gdf)

# %% Plot Bands together

# SAR Features
plot_multi_bands(input_image_dir / 'input_image.tif',
                  plot_dir, 
                  ['VV_PRE', 'VH_PRE', 'VV_VH_RATIO_PRE',
                    'VV_POST', 'VH_POST', 'VV_VH_RATIO_POST',
                    'VV_CHANGE', 'VH_CHANGE', 'VV_VH_RATIO_CHANGE'], # None to plot all
                  custom_extent='square', 
                  figsize=(18, 18),
                  output_filename='SAR_Features.png'
                  );
# %%
# Contextual Features
plot_multi_bands(input_image_dir / 'input_image.tif',
                  plot_dir, 
                  ['SLOPE', 'HAND', 'EDTW', 'LAND_COVER'], # None to plot all
                  custom_extent='square', 
                  figsize=(12, 12),
                  n_cols=2,
                  output_filename='Contextual_Features.png'
                  );
# %%
# Input Data and intermediate products
plot_multi_bands(input_image_dir / 'input_image.tif',
                  plot_dir, 
                  ['DEM', 'BURNED_DEM', 'OSM_WATER', 'LABEL'], # None to plot all
                  custom_extent='square', 
                  figsize=(12, 12),
                  n_cols=2,
                  output_filename='Input_Data_and_Intermediate_Products.png'
                  );

# %% Plot Bands individually
plot_bands(input_image_dir / 'input_image.tif',
            plot_dir,
            custom_extent='square',
            bands_to_plot=['OSM_WATER', 'BURNED_DEM', 'HAND', 'DEM'] # None to plot all
            )

# %% Plot 3D Map Stack
plot_image_stack(
    input_image_dir / 'input_image.tif', 
    plot_dir,
    ['VV_PRE', 'VH_PRE', 'VV_VH_RATIO_PRE',
                    'VV_POST', 'VH_POST', 'VV_VH_RATIO_POST',
                    'VV_CHANGE', 'VH_CHANGE', 'VV_VH_RATIO_CHANGE',
                    'SLOPE', 'HAND', 'EDTW', 'LAND_COVER'],
    spacing=1.5,
    horizontal=False
)

# %%

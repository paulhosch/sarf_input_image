# %% Imports
from pathlib import Path
from input_image.vis.plot_band import plot_bands
from input_image.vis.plot_image_stack import plot_image_stack
from input_image.vis.plut_multi_band import plot_multi_bands

# %% Set paths (edit these as needed for your case study)
site_id = 'valencia'
data_dir = '../../data'
case_study_dir = Path(data_dir) / 'case_studies' / site_id
input_image_dir = case_study_dir / 'input_image'
plot_dir = input_image_dir / 'plots'

# %% Plot SAR Features
plot_multi_bands(input_image_dir / 'input_image.tif',
                  plot_dir, 
                  ['VV_PRE', 'VH_PRE', 'VV_VH_RATIO_PRE',
                    'VV_POST', 'VH_POST', 'VV_VH_RATIO_POST',
                    'VV_CHANGE', 'VH_CHANGE', 'VV_VH_RATIO_CHANGE'],
                  custom_extent='square', 
                  figsize=(18, 18),
                  output_filename='SAR_Features.png'
                  )

# %% Plot Contextual Features
plot_multi_bands(input_image_dir / 'input_image.tif',
                  plot_dir, 
                  ['SLOPE', 'HAND', 'EDTW', 'LAND_COVER'],
                  custom_extent='square', 
                  figsize=(12, 12),
                  n_cols=2,
                  output_filename='Contextual_Features.png'
                  )

# %% Plot Input Data and Intermediate Products
plot_multi_bands(input_image_dir / 'input_image.tif',
                  plot_dir, 
                  ['DEM', 'BURNED_DEM', 'OSM_WATER', 'LABEL'],
                  custom_extent='square', 
                  figsize=(12, 12),
                  n_cols=2,
                  output_filename='Input_Data_and_Intermediate_Products.png'
                  )

# %% Plot Bands Individually
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
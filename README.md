# saRFlood-1 Input Image

This repository contains functionalities for the 1. part of the saRFlood pipeline - Generating the `input_image.tif`.

The `input_image.tif` has bands for input_features, label and more, these will later be used as data for sampling, prediction and validation.

## Input Image Bands

Throught the interactive pipeline.py skript the following bands will be computed and added to the image.

### SAR Features

### Contextual Features

### Other Bands

// images

## Output

The localy saved `input_image.tif` has a resolution of `10m` (DEMs original 30m resolution is resampled) and is defined in `ESPG:4326`
Individual rasters (intermediate results) are saved in `input_image\bands\` in there original reasolutionand projection and serve as a "cache".

## Usage

The usage is outlined in the interactive `pipeline.py`

1. Authenticate with Google Earth Engine:

```bash
earthengine authenticate
```

The main interactive pipeline script `pipeline.py` requires minimal user input:

2. Define the case study name
3. Provide AOI shapefile
4. Provide ground truth shapefile (optional if you want the label band for training or validation)

## Data Sources

- **Sentinel-1 SAR**: Pre/post-event backscatter [GEE API] `COPERNICUS/S1_GRD`

  - Source: European Union/ESA/Copernicus
  - Resolution: 10m

- **FathomDEM**: Global terrain map [Zenodo API]

  - Source: Uhe, P., Lucas, C., Hawker, L., et al. (2025). FathomDEM: an improved global terrain map using a hybrid vision transformer model. Environmental Research Letters, 20(3).
  - Resolution: 30m

- **ESRI Land Cover**: Global land cover classification [GEE API] `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`

  - Source: Karra, K., Kontgis, C., et al. (2021). Global land use/land cover with Sentinel-2 and deep learning. IGARSS 2021
  - Resolution: 10m

- **OpenStreetMap**: Water features [OSM API]
  - Source: © OpenStreetMap contributors
  - Features:Rivers, lakes, and other water bodies

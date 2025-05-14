# saRFlood-1 Input Image

This repository contains functionalities for the 1. part of the saRFlood pipeline - Generating the `input_image.tif`.

The `input_image.tif` has bands for input_features, label and more, these will later be used as data for sampling, prediction and validation.

## Input Image Bands

Throught the interactive pipeline.py skript the following bands will be computed and added to the image.

### SAR Features

![SAR_Features](https://github.com/user-attachments/assets/bf817df2-dc06-42fa-b69b-b9e5a003be1c)

### Contextual Features

![Contextual_Features](https://github.com/user-attachments/assets/52415ca6-3e2a-4fa7-a717-90c5c51a841d)

### Other Bands

![Input_Data_and_Intermediate_Products](https://github.com/user-attachments/assets/66d78f1a-0ca6-4d0d-86f5-fa9a21dd5a11)

## Output

The localy saved `input_image.tif` has a resolution of `10m` (DEMs original 30m resolution is resampled) and is defined in `ESPG:4326`
Individual rasters (intermediate results) are saved in `input_image\bands\` in there original reasolutionand projection and serve as a "cache".

## Installation & Usage

1. **Clone the repository**

```bash
git clone https://github.com/paulhosch/sarf_input_image.git
cd sarf_input_image
```

2. **Create and activate a conda environment with Python 3.10**

```bash
conda create -n sarf_input_image python=3.10 -y
conda activate sarf_input_image
```

3. **Install GDAL and richdem (required for DEM processing)**

```bash
conda install -c conda-forge gdal richdem
```

4. **Install requirements**

```bash
pip install -r requirements.txt
```

5. **(Optional) Register the environment as a Jupyter kernel**

```bash
pip install ipykernel
python -m ipykernel install --user --name sarf_input_image --display-name "Python (sarf_input_image)"
```

6. **Authenticate with Google Earth Engine if required**

```bash
earthengine authenticate
```

---

## Installation & Usage

1. **Clone the repository**

```bash
git clone https://github.com/paulhosch/sarf_input_image.git
cd sarf_input_image
```

2. **Create and activate a conda environment with Python 3.10**

```bash
conda create -n sarf_input_image python=3.10 -y
conda activate sarf_input_image
```

3. **Install GDAL and richdem (required for DEM processing)**

```bash
conda install -c conda-forge gdal richdem
```

4. **Install requirements**

```bash
pip install -r requirements.txt
```

5. **(Optional) Register the environment as a Jupyter kernel**

```bash
pip install ipykernel
python -m ipykernel install --user --name sarf_input_image --display-name "Python (sarf_input_image)"
```

6. **Authenticate with Google Earth Engine if required**

```bash
earthengine authenticate
```

---

### Troubleshooting: NumPy Version Conflicts

Some geospatial packages (like GDAL or richdem) may upgrade NumPy to version 2.x, which is not compatible with all packages (e.g., cmocean, pysheds). If you get errors about missing `np.unicode_` or similar, force NumPy 1.26.4 with:

```bash
conda install numpy=1.26.4
```

## Running the Pipeline

The main interactive pipeline script is `pipeline.py`. Run it in an interactive Python environment (e.g., Jupyter, VSCode interactive window, or with `ipython`).

**Minimal user input required:**

- Define the case study name
- Provide AOI shapefile
- (Optional) Provide ground truth shapefile for label band (training/validation)

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

## Note on Coastal AOIs

If your Area of Interest (AOI) is located near the coast, you must download global seawater polygons from [OpenStreetMap Water Polygons](https://osmdata.openstreetmap.de/data/water-polygons.html). Place the downloaded `water_polygons.shp` and associated files in a `global_seawater` folder inside your `data/` directory.

Example:

```
data/
└── global_seawater/
    ├── water_polygons.shp
    ├── water_polygons.dbf
    ├── water_polygons.shx
    └── ...
```

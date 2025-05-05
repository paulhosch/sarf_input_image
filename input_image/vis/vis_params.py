"""
Visualization parameters for the input image bands.
"""
import matplotlib.pyplot as plt
import numpy as np
from cmocean import cm
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# Truncate to exclude blue (bottom ~50%)
TOPO_COPPED = truncate_colormap(cm.topo, minval=0.5, maxval=1.0)

# Truncate to include blue (but only for the bottom 2% of the colormap)
def truncate_colormap_blue(cmap, minval=0.0, maxval=0.02, n=256):
    # Get the original colormap colors
    colors = cmap(np.linspace(0, 1, n))
    
    # Create a new colormap with blue at the bottom 2%
    blue_colors = np.array([[0, 0, 1, 1]])  # Pure blue with alpha=1
    main_colors = colors[int(n*maxval):]
    
    # Combine blue and main colors
    new_colors = np.vstack((blue_colors, main_colors))
    
    # Create new colormap
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc_blue({cmap.name})',
        new_colors
    )
    return new_cmap

TOPO_CROPPED_BLUE = truncate_colormap_blue(cm.topo, minval=0.0, maxval=0.02)

TOPO_UNCOPPED = truncate_colormap(cm.topo, minval=0.0, maxval=0.02)

CORD_LABEL_SIZE = 16
TITLE_SIZE = 18
CBAR_LABEL_SIZE = 16

# Visualization parameters for each band type
SAR_DERIVATIVE_CMAP = "cividis"
VIS_PARAMS = {
    "DEM": {
        "cmap": TOPO_COPPED,  # Using the cropped topo colormap
        "vmin": None,
        "vmax": None,
        "title": "FathomDEM",
        "cbar_label": "Elevation (m)",
        "continuous": True
    },
    "BURNED_DEM": {
        "cmap": TOPO_COPPED,  # Using the cropped topo colormap
        "vmin": None,
        "vmax": None,
        "title": "Stream-Burned DEM",
        "cbar_label": "Elevation (m)",
        "continuous": True
    },
    "SLOPE": {
        "cmap": LinearSegmentedColormap.from_list("slope_cmap", ["white", "#D45E00"]),
        "vmin": 0,
        "vmax": 90,  
        "title": "Horton's Slope",
        "cbar_label": "Slope (°)", 
        "continuous": True
    },
    "OSM_WATER": {
        "cmap": "Blues",
        "vmin": 0,
        "vmax": 1,
        "title": "OSM Water Features",
        "cbar_label": "Water presence",
        "continuous": False,
        "colors": ["#C99060", "#0173B2"],
        "class_names": ["No Water", "Water"]
    },
    "EDTW": {
        "cmap": "cividis",
        "vmin": None,
        "vmax": None,
        "title": "Euclidean Distance to Water",
        "cbar_label": "Distance (m)",
        "continuous": True
    },
    "HAND": {
        "cmap": TOPO_COPPED,
        "vmin": None,
        "vmax": None,
        "title": "Height Above Nearest Drainage",
        "cbar_label": "HAND (m)",
        "continuous": True
    },
    "VV_PRE": {
        "cmap": "Greys_r",
        "vmin": -30,
        "vmax": 0,
        "title": "Pre-Event VV",
        "cbar_label": r"$\sigma^0$ (dB)",
        "continuous": True
    },
    "VH_PRE": {
        "cmap": "Greys_r",
        "vmin": -30,
        "vmax": 0,
        "title": "Pre-Event VH",
        "cbar_label": r"$\sigma^0$ (dB)",
        "continuous": True
    },
    "VV_POST": {
        "cmap": "Greys_r",
        "vmin": -30,
        "vmax": 0,
        "title": "Post-Event VV",
        "cbar_label": r"$\sigma^0$ (dB)",
        "continuous": True
    },
    "VH_POST": {
        "cmap": "Greys_r",
        "vmin": -30,
        "vmax": 0,
        "title": "Post-Event VH",
        "cbar_label": r"$\sigma^0$ (dB)",
        "continuous": True
    },
    "VV_CHANGE": {
        "cmap": SAR_DERIVATIVE_CMAP,
        "vmin": None,
        "vmax": None,
        "title": "VV Change",
        "cbar_label": "Post/Pre (-)",
        "continuous": True
    },
    "VH_CHANGE": {
        "cmap": SAR_DERIVATIVE_CMAP,
        "vmin": None,
        "vmax": None,
        "title": "VH Change",
        "cbar_label": "Post/Pre (-)",
        "continuous": True
    },
    "VV_VH_RATIO_PRE": {
        "cmap": SAR_DERIVATIVE_CMAP,
        "vmin": None,
        "vmax": None,
        "title": "Pre-event VV/VH ",
        "cbar_label": "Ratio (-)",
        "continuous": True
    },
    "VV_VH_RATIO_POST": {
        "cmap": SAR_DERIVATIVE_CMAP,
        "vmin": None,
        "vmax": None,
        "title": "Post-event VV/VH ",
        "cbar_label": "Ratio (-)",
        "continuous": True
    },
    "VV_VH_RATIO_CHANGE": {
        "cmap": SAR_DERIVATIVE_CMAP,
        "vmin": None,
        "vmax": None,
        "title": "VV/VH Change",
        "cbar_label": "Ratio (-)",
        "continuous": True
    },
    "LAND_COVER": {
        "cmap": "custom",  # This will be overridden by colors list
        "vmin": 0,
        "vmax": 11,  # Highest ESRI class value
        "title": "ESRI 2023 Land Cover",
        "cbar_label": "Class",
        "continuous": False,
        "colors": [
            "#FFFFFF",  # No data (0)
            "#0173B2",  # Water (1)
            "#009E73",  # Trees (2)
            "#FFFFFF",  # Not used (3)
            "#56B4E9",  # Flooded Vegetation (4)
            "#ECE034",  # Crops (5)
            "#FFFFFF",  # Not used (6)
            "#949494",  # Built Area (7)
            "#EDE9E4",  # Bare Ground (8)
            "#F2FAFF",  # Snow/Ice (9)
            "#C8C8C8",  # Clouds (10)
            "#C99060"   # Rangeland (11)
        ],
        "class_names": [
            "",
            "Water",
            "Trees",
            "", 
            "Flooded Vegetation",
            "Crops",
            "",
            "Built Area",
            "Bare Ground",
            "Snow/Ice",
            "",
            "Rangeland"
        ]
    },
    "LABEL": {
        "cmap": "RdBu",
        "vmin": 0,
        "vmax": 1,
        "title": "Flood Label",
        "cbar_label": "Flood presence",
        "continuous": False,
        "colors": ["#C99060", "#56B4E9"],
        "class_names": ["No Flood", "Flood"]
    }
}

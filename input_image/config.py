"""
Configuration settings for the input image pipeline.
"""

import numpy as np
import os
from pathlib import Path

# File type constants
GEOTIFF_EXT = ".tif"

# Universal raster settings
UNIVERSAL_DTYPE = "float32"
UNIVERSAL_CRS = "EPSG:4326"
UNIVERSAL_NODATA = np.nan

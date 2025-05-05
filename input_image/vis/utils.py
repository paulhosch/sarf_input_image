from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import math
from .vis_params import VIS_PARAMS, CORD_LABEL_SIZE



def add_grid_lines(ax, linewidth=0.5, top_labels=False, right_labels=False, bottom_labels=True, left_labels=True):
    """
    Add gridlines with formatted latitude and longitude labels to a cartopy map.
    
    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The cartopy axes to add gridlines to
    """
    gl = ax.gridlines(draw_labels=True, linewidth=linewidth, alpha=0.5, linestyle='--')
    gl.top_labels = top_labels
    gl.right_labels = right_labels
    gl.bottom_labels = bottom_labels
    gl.left_labels = left_labels
    gl.xformatter = LongitudeFormatter(number_format='.1f')
    gl.yformatter = LatitudeFormatter(number_format='.1f')
    gl.xlocator = mticker.LinearLocator(3) 
    gl.ylocator = mticker.LinearLocator(3)

    # Increase font size and rotate labels
    gl.xlabel_style = {'size': CORD_LABEL_SIZE}
    gl.ylabel_style = {'size': CORD_LABEL_SIZE, 'rotation': 90}
    
def get_square_extent(bounds):
    """
    Create a square extent centered on the input bounds.
    
    This ensures that the map has equal dimensions in both directions,
    using the larger of the width or height as the common dimension.
    
    Parameters:
    -----------
    bounds : tuple
        (left, bottom, right, top) boundaries in map units
        
    Returns:
    --------
    list
        [left, right, bottom, top] extent for cartopy's set_extent method
    """
    # Create square bounds by finding the larger dimension
    # bounds = (left, bottom, right, top)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    
    # Use the larger dimension to create a square
    max_dimension = max(width, height)
    square_bounds = [
        center_x - max_dimension / 2,  # left
        center_y - max_dimension / 2,  # bottom
        center_x + max_dimension / 2,  # right
        center_y + max_dimension / 2   # top
    ]
    square_extent = [square_bounds[0], square_bounds[2], square_bounds[1], square_bounds[3]]  # Square extent
    return square_extent

def get_extend_dimension(map_extent):
        # Calculate extent dimensions in kilometers
    lon_diff = abs(map_extent[1] - map_extent[0])  # width in degrees
    lat_diff = abs(map_extent[3] - map_extent[2])  # height in degrees
    
    # Calculate midpoint latitude for more accurate conversion
    mid_lat = (map_extent[2] + map_extent[3]) / 2
    
    # Convert degrees to kilometers
    # 1 degree of latitude is approximately 111 km
    # 1 degree of longitude varies with latitude
    km_per_lon_degree = 111 * math.cos(math.radians(mid_lat))
    km_per_lat_degree = 111
    
    # Calculate dimensions in km
    width_km = lon_diff * km_per_lon_degree
    height_km = lat_diff * km_per_lat_degree
    
    # Format the width dimension for display
    if width_km < 1:
        width_m = width_km * 1000
        extent_dimensions = f"{width_m:.0f} m"
    elif width_km < 10:
        extent_dimensions = f"{width_km:.1f} km" 
    else:
        extent_dimensions = f"{width_km:.0f} km"
    return extent_dimensions

def get_map_extent(custom_extent, bounds):
    # Get data extent (original bounds)
    data_extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    
    # Determine which extent to use based on custom_extent parameter
    if custom_extent is None:
        # Use original data extent
        map_extent = data_extent
        print(f"Using original data extent: {map_extent}")
    elif custom_extent == 'square':
        # Use square extent centered on data
        map_extent = get_square_extent(bounds)
        print(f"Using square extent: {map_extent}")
    elif isinstance(custom_extent, (list, tuple)) and len(custom_extent) == 4:
        # Use user-specified extent: [left, right, bottom, top]
        map_extent = custom_extent
        print(f"Using custom extent: {map_extent}")
    else:
        # Invalid custom_extent value, fall back to data extent
        map_extent = data_extent
        print(f"Invalid custom_extent '{custom_extent}', using data extent: {map_extent}") 
    
    extent_dimensions = get_extend_dimension(map_extent)
    print(f"Extent dimensions: {extent_dimensions}")
    return data_extent, map_extent, extent_dimensions
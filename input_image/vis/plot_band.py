"""
Visualization utilities for the input image pipeline.
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path
from .vis_params import VIS_PARAMS, TITLE_SIZE, CBAR_LABEL_SIZE
from .utils import add_grid_lines, get_map_extent

def find_best_band_match(description, vis_params_keys):
    """
    Find matching band name in vis_params for a given band description.
    Uses direct case-insensitive matching.
    
    Args:
        description: Band description from the raster file
        vis_params_keys: List of keys from the VIS_PARAMS dictionary
        
    Returns:
        The matching band name or None if no match found
    """
    description_upper = description.upper()
    
    # Direct match (case insensitive)
    for band_name in vis_params_keys:
        if band_name.upper() == description_upper:
            return band_name
    
    return None

def plot_bands(input_image_path, output_dir, bands_to_plot=None, custom_extent=None, figsize=(10, 10), plot_cords=True, plot_cbar = True):
    """
    Visualize all bands from an input image.
    
    Args:
        input_image_path: Path to the input image file
        output_dir: Directory to save visualizations
        custom_extent : None, 'square', or list, optional
        Controls the map extent:
        - None: use original data extent
        - 'square': use square extent centered on data
        - list [left, right, bottom, top]: use specified extent in degrees
        bands_to_plot: List of band names to plot (default: None = plot all bands)
    """
    input_image_path = Path(input_image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert band names to uppercase if provided
    if bands_to_plot is not None:
        bands_to_plot = [band.upper() for band in bands_to_plot]
    
    # Open the input image
    with rasterio.open(input_image_path) as src:
        num_bands = src.count
        print(f"Input image has {num_bands} bands")
        
        # Process each band in the input image
        for band_idx in range(num_bands):

            # Get band description
            description = src.descriptions[band_idx] if band_idx < len(src.descriptions) else f"Band {band_idx+1}"

            # Use the description to find a matching band name in the vis_params
            matched_band = find_best_band_match(description, VIS_PARAMS.keys())
            
            # Skip this band if bands_to_plot is specified and this band is not in the list
            if bands_to_plot is not None:
                # Check if we should plot this band
                should_plot = False
                description_upper = description.upper()
                
                for band_name in bands_to_plot:
                    # Simple direct matching
                    if (matched_band and band_name == matched_band.upper()) or band_name == description_upper:
                        should_plot = True
                        break
                
                if not should_plot:
                    continue  # Skip this band
                    
            # If no match found, use default parameters
            if matched_band is None:
                print(f"No visualization parameters found for band {band_idx+1}, using defaults")
                vis_params = {
                    "cmap": "viridis",
                    "vmin": None,
                    "vmax": None,
                    "title": description,
                    "cbar_label": f"Value",
                    "continuous": True
                }
            else:
                print(f"Using visualization parameters for '{matched_band}' for band with description: '{description}'")
                vis_params = VIS_PARAMS[matched_band]
            
            # Read the band data
            data = src.read(band_idx + 1)  # rasterio uses 1-based indexing

            
            # Set up the figure with cartopy
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=ccrs.PlateCarree())

            # Set the map extend
            data_extent, map_extent, extent_dimensions = get_map_extent(custom_extent, src.bounds)
            ax.set_extent(map_extent)
            ax.set_aspect('equal')

            # Compute reasonable vmin/vmax if not specified
            if vis_params["vmin"] is None:
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    vmin = np.percentile(valid_data, 2)
                else:
                    vmin = 0
            else:
                vmin = vis_params["vmin"]
                
            if vis_params["vmax"] is None:
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    vmax = np.percentile(valid_data, 98)
                else:
                    vmax = 1
            else:
                vmax = vis_params["vmax"]
            
            # Create custom colormap for categorical data
            cmap = vis_params["cmap"]
            if not vis_params["continuous"] and "colors" in vis_params:
                cmap = ListedColormap(vis_params["colors"])
            
            # Plot the raster data
            img = ax.imshow(
                data, 
                origin='upper',
                extent=data_extent,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            # Add gridlines with lat/lon labels
            if plot_cords:
                add_grid_lines(ax)
            
            # Add colorbar for continuous data, or legend for categorical data
            if plot_cbar:
                if vis_params["continuous"]:
                    # Add a horizontal colorbar below the plot with consistent spacing
                    cbar = plt.colorbar(img, ax=ax, pad=0.05, fraction=0.046, orientation='horizontal')
                    cbar.set_label(vis_params.get("cbar_label", "Value"), fontsize=CBAR_LABEL_SIZE)
                    
                    # Set 5 ticks including min and max for continuous data
                    ticks = np.linspace(vmin, vmax, 5)
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels([f"{tick:.0f}" for tick in ticks], fontsize=CBAR_LABEL_SIZE)
                else:
                    # Create a legend with patches for categorical data
                    if "class_names" in vis_params and "colors" in vis_params:
                        class_names = vis_params["class_names"]
                        colors = vis_params["colors"]
                        
                        # Create a list of legend handles (colored patches)
                        legend_patches = []
                        
                        # Determine if we need to map values to indices
                        if "class_values" in vis_params:
                            # Use explicit mapping between values and class names
                            class_values = vis_params["class_values"]
                            for i, (name, value) in enumerate(zip(class_names, class_values)):
                                if name and i < len(colors):  # Skip empty class names
                                    color = colors[i]
                                    patch = mpatches.Patch(color=color, label=name)
                                    legend_patches.append(patch)
                        else:
                            # Assume class indices are sequential (0, 1, 2...)
                            for i, name in enumerate(class_names):
                                if name and i < len(colors):  # Skip empty class names
                                    color = colors[i]
                                    patch = mpatches.Patch(color=color, label=name)
                                    legend_patches.append(patch)
                        
                        # Add the legend below the plot with consistent spacing
                        if legend_patches:
                            legend = ax.legend(
                                handles=legend_patches,
                                loc='upper center', 
                                bbox_to_anchor=(0.5, -0.05),  # Adjusted to match colorbar spacing
                                ncol=min(4, len(legend_patches)),
                                fontsize=CBAR_LABEL_SIZE
                            )
                            legend.get_frame().set_facecolor('none')
                            legend.get_frame().set_linewidth(0)
                    else:
                        # Fallback for categorical data without class_names
                        cbar = plt.colorbar(img, ax=ax, pad=0.15, fraction=0.046, orientation='horizontal')
                        cbar.set_label(vis_params.get("cbar_label", "Categories"), fontsize=CBAR_LABEL_SIZE)
            
            # Set title and save figure
            plt.title(vis_params["title"], fontsize=TITLE_SIZE, pad=10, weight='bold')
            plt.tight_layout()
            
            # Extract a short name for the file from the description or band index
            if matched_band:
                filename = f"{matched_band}.png"
            else:
                filename = f"band{band_idx+1}.png"
                
            fig_path = output_dir / filename
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"{description} plot saved to {fig_path}")
    
    print(f"\nAll visualizations saved to {output_dir}")


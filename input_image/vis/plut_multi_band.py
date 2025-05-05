"""
Visualization utilities for plotting multiple bands in one figure.
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
import math
from .vis_params import VIS_PARAMS, TITLE_SIZE, CBAR_LABEL_SIZE
from .utils import add_grid_lines, get_map_extent
from .plot_band import find_best_band_match  # Reuse the band matching function

def plot_multi_bands(input_image_path, output_dir, bands_to_plot, output_filename=None, 
                    layout=None, figsize=None, custom_extent=None, dpi=300, n_cols=3):
    """
    Plot multiple bands from an input image as subplots in a single figure.
    
    Args:
        input_image_path : str or Path
            Path to the input image file
        output_dir : str or Path
            Directory to save visualization
        bands_to_plot : list
            List of band names or descriptions to plot
        output_filename : str, optional
            Name for the output image file (default: 'multi_band.png')
        layout : tuple, optional
            Number of rows and columns for the subplot layout (default: auto-calculated)
        figsize : tuple, optional
            Figure size in inches (width, height) (default: auto-calculated)
        custom_extent : None, 'square', or list, optional
            Controls the map extent:
            - None: use original data extent
            - 'square': use square extent centered on data
            - list [left, right, bottom, top]: use specified extent in degrees
        dpi : int, optional
            Resolution for saved figure (default: 300)
        n_cols : int, optional
            Number of columns in the subplot grid (default: 3)
            
    Returns:
        matplotlib.figure.Figure
            The created figure object
    """
    # Validate and convert inputs
    input_image_path = Path(input_image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not output_filename:
        output_filename = 'multi_band.png'
    
    # Determine layout
    n_bands = len(bands_to_plot)
    if layout is None:
        # Calculate a reasonable layout based on number of bands
        n_cols = min(n_cols, n_bands)  # Use specified number of columns, but not more than bands
        n_rows = math.ceil(n_bands / n_cols)
        layout = (n_rows, n_cols)
    else:
        n_rows, n_cols = layout
    
    # Determine figure size if not provided
    if figsize is None:
        # Base size per subplot
        subplot_width, subplot_height = 6, 6
        figsize = (subplot_width * n_cols, subplot_height * n_rows)
    
    # Create figure and axes with cartopy projection
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=figsize, 
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=False  # Use gridspec instead
    )
    # Add spacing parameters for the subplots
    plt.subplots_adjust(
       # wspace=0.1,    # width spacing between subplots
       hspace=0.5,   # height spacing between subplots
       # top=0.9,       # top boundary (space for title)
       # bottom=0.25,   # bottom boundary (space for legend/colorbar)
        #left=0.05,     # left boundary
    )
    
    # Handle single subplot case
    if n_bands == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Create separate axes for all colorbars/legends to ensure consistent spacing
    cbar_axes = []
    for i in range(len(axes)):
        # Create colorbar/legend axes below each plot with consistent position
        cax = fig.add_axes([0, 0, 0.1, 0.1])  # Placeholder, will be updated later
        cbar_axes.append(cax)
    
    # Open the input image
    with rasterio.open(input_image_path) as src:
        num_bands = src.count
        print(f"Input image has {num_bands} bands")
        
        # Get consistent map extent for all plots
        data_extent, map_extent, extent_dimensions = get_map_extent(custom_extent, src.bounds)
        
        # Track which band names we've successfully found and plotted
        plotted_bands = []
        plot_idx = 0
        
        # Convert requested band names to uppercase
        bands_to_plot = [band.upper() for band in bands_to_plot]
        
        # First pass: find band indices that match the requested bands
        band_mapping = {}
        for band_idx in range(num_bands):
            description = src.descriptions[band_idx] if band_idx < len(src.descriptions) else f"Band {band_idx+1}"
            description_upper = description.upper()
            matched_band = find_best_band_match(description, VIS_PARAMS.keys())
            
            # Check if this band should be plotted
            for band_name in bands_to_plot:
                # Simple direct matching
                if ((matched_band and band_name == matched_band.upper()) or 
                    band_name == description_upper):
                    band_mapping[band_name] = {
                        'band_idx': band_idx,
                        'description': description,
                        'matched_band': matched_band
                    }
                    break
        
        # Second pass: plot bands in the requested order
        for i, band_name in enumerate(bands_to_plot):
            if band_name in band_mapping:
                band_info = band_mapping[band_name]
                band_idx = band_info['band_idx']
                description = band_info['description']
                matched_band = band_info['matched_band']
                
                # Get current axis
                ax = axes[i]
                
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
                
                # Set map extent
                ax.set_extent(map_extent)
                ax.set_aspect('equal')
                
                # Add gridlines with lat/lon labels only for the last subplot
                is_last_subplot = (i == len(bands_to_plot) - 1)
                if is_last_subplot:
                    add_grid_lines(ax, linewidth=0, top_labels=False, right_labels=True, bottom_labels=True, left_labels=False)
                
                # Compute reasonable vmin/vmax if not specified
                if vis_params["vmin"] is None:
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        vmin = np.percentile(valid_data, 2)
                        # Better rounding approach for nice numbers
                        if abs(vmin) < 1e-10:
                            vmin = 0
                        elif abs(vmin) < 1:
                            # For small values, round to 2 decimal places
                            vmin = np.floor(vmin * 100) / 100
                        elif abs(vmin) < 10:
                            # For values between 1 and 10, round to 1 decimal place
                            vmin = np.floor(vmin * 10) / 10
                        else:
                            # For larger values, round to nearest half order of magnitude
                            magnitude = 10 ** (np.floor(np.log10(abs(vmin))))
                            half_magnitude = magnitude / 2
                            vmin = np.floor(vmin / half_magnitude) * half_magnitude
                    else:
                        vmin = 0
                else:
                    vmin = vis_params["vmin"]
                    
                if vis_params["vmax"] is None:
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        vmax = np.percentile(valid_data, 98)
                        # Better rounding approach for nice numbers
                        if abs(vmax) < 1e-10:
                            vmax = 0.01  # Avoid zero for vmax
                        elif abs(vmax) < 1:
                            # For small values, round to 2 decimal places
                            vmax = np.ceil(vmax * 100) / 100
                        elif abs(vmax) < 10:
                            # For values between 1 and 10, round to 1 decimal place
                            vmax = np.ceil(vmax * 10) / 10
                        else:
                            # For larger values, round to nearest half order of magnitude
                            magnitude = 10 ** (np.floor(np.log10(abs(vmax))))
                            half_magnitude = magnitude / 2
                            vmax = np.ceil(vmax / half_magnitude) * half_magnitude
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
                
                # Add colorbar for continuous data, or legend for categorical data
                if vis_params["continuous"]:
                    # Calculate consistent position for colorbar axes
                    cbar_height = 0.015
                    cbar_offset = 0.035
                    # Get the position of the current subplot
                    pos = axes[i].get_position()
                    # Position the colorbar axes centered below the subplot
                    cbar_axes[i].set_position([pos.x0, pos.y0 - cbar_offset, pos.width, cbar_height])
                    
                    # Add a horizontal colorbar below the plot
                    cbar = fig.colorbar(img, cax=cbar_axes[i], orientation='horizontal', pad=0)
                    cbar.set_label(vis_params.get("cbar_label", "Value"), fontsize=CBAR_LABEL_SIZE)
                    
                    # Set 5 ticks including min and max for continuous data
                    ticks = np.linspace(vmin, vmax, 5)
                    cbar.set_ticks(ticks)
                    
                    # Format tick labels based on value range
                    if vmax < 5:
                        cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks], fontsize=CBAR_LABEL_SIZE)
                    else:
                        cbar.set_ticklabels([f"{tick:.0f}" for tick in ticks], fontsize=CBAR_LABEL_SIZE)
                else:
                    # Hide the unused colorbar axis
                    cbar_axes[i].set_visible(False)
                    
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
                            for j, (name, value) in enumerate(zip(class_names, class_values)):
                                if name and j < len(colors):  # Skip empty class names
                                    color = colors[j]
                                    patch = mpatches.Patch(color=color, label=name)
                                    legend_patches.append(patch)
                        else:
                            # Assume class indices are sequential (0, 1, 2...)
                            for j, name in enumerate(class_names):
                                if name and j < len(colors):  # Skip empty class names
                                    color = colors[j]
                                    patch = mpatches.Patch(color=color, label=name)
                                    legend_patches.append(patch)
                        
                        # Add the legend directly to the main axis but with consistent positioning
                        if legend_patches:
                            legend = axes[i].legend(
                                handles=legend_patches,
                                loc='upper center', 
                                bbox_to_anchor=(0.5, -0.08),  # Consistent position for all legends
                                ncol=min(2, len(legend_patches)),
                                fontsize=CBAR_LABEL_SIZE-2
                            )
                            legend.get_frame().set_facecolor('none')
                            legend.get_frame().set_linewidth(0)
                    else:
                        # Fallback for categorical data without class_names
                        # Calculate consistent position for colorbar axes
                        pos = axes[i].get_position()
                        cbar_axes[i].set_position([pos.x0, pos.y0 - 0.13, pos.width, 0.03])
                        
                        cbar = fig.colorbar(img, cax=cbar_axes[i], orientation='horizontal')
                        cbar.set_label(vis_params.get("cbar_label", "Categories"), fontsize=CBAR_LABEL_SIZE)
                
                # Set title
                ax.set_title(vis_params["title"], fontsize=TITLE_SIZE, pad=10, weight='bold')
                plotted_bands.append(description)
                plot_idx += 1
            else:
                print(f"Warning: Band '{band_name}' not found in the input image")
        
        # Hide any unused subplots
        for j in range(plot_idx, len(axes)):
            axes[j].set_visible(False)
        
        # Save the figure
        output_path = output_dir / output_filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Multi-band plot saved to {output_path}")
        
        return fig

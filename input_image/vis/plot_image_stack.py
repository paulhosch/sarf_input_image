"""
Visualization utility for creating 3D stacks of multiple image bands.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import rasterio
from mpl_toolkits.mplot3d import Axes3D

from .vis_params import VIS_PARAMS

def plot_image_stack(input_image_path, output_dir, bands_to_plot, spacing=1.0, output_filename=None, horizontal=True):
    """
    Visualize multiple bands from an input image as a simplified 3D stack.
    
    Args:
        input_image_path: Path to the input image file
        output_dir: Directory to save visualization
        bands_to_plot: List of band names to include in the stack
        spacing: Spacing between layers (default: 1.0)
        output_filename: Custom filename for the output plot (default: None, will use 'image_stack_3d.png')
        horizontal: Whether to stack images horizontally (True) or vertically (False)
    """
    input_image_path = Path(input_image_path)
    output_dir = Path(output_dir)
    
    stack_type = "horizontal" if horizontal else "vertical"
    print(f"Creating 3D {stack_type} stack visualization of bands: {', '.join(bands_to_plot)}")
    print(f"Plot will be saved to: {output_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert band names to uppercase
    bands_to_plot = [band.upper() for band in bands_to_plot]
    
    # Open the input image
    with rasterio.open(input_image_path) as src:
        num_bands = src.count
        
        # Get geographic bounds
        left, bottom, right, top = src.bounds
        x = np.linspace(left, right, src.width)
        y = np.linspace(bottom, top, src.height)
        X, Y = np.meshgrid(x, y)
        
        # Find band indices and data for each requested band
        band_data = []
        vis_params_list = []
        band_names_found = []
        
        for band_name in bands_to_plot:
            found = False
            
            # Find band index based on name
            for band_idx in range(num_bands):
                description = src.descriptions[band_idx] if band_idx < len(src.descriptions) else f"Band {band_idx+1}"
                description_upper = description.upper()
                
                # Simple direct matching
                if band_name == description_upper:
                    # Read the band data
                    data = src.read(band_idx + 1)  # rasterio uses 1-based indexing
                    
                    # Find matching visualization parameters (direct match)
                    matched_band = None
                    for param_name in VIS_PARAMS.keys():
                        if param_name.upper() == band_name:
                            matched_band = param_name
                            break
                    
                    # Use default parameters if no match is found
                    if matched_band is None:
                        print(f"No visualization parameters found for band '{band_name}', using defaults")
                        vis_params = {
                            "cmap": "viridis",
                            "vmin": None,
                            "vmax": None,
                            "continuous": True
                        }
                    else:
                        vis_params = VIS_PARAMS[matched_band]
                    
                    # Compute reasonable vmin/vmax if not specified
                    if vis_params["vmin"] is None:
                        valid_data = data[~np.isnan(data)]
                        if len(valid_data) > 0:
                            vis_params["vmin"] = np.percentile(valid_data, 2)
                        else:
                            vis_params["vmin"] = 0
                            
                    if vis_params["vmax"] is None:
                        valid_data = data[~np.isnan(data)]
                        if len(valid_data) > 0:
                            vis_params["vmax"] = np.percentile(valid_data, 98)
                        else:
                            vis_params["vmax"] = 1
                    
                    # Normalize data for 3D visualization
                    norm_data = (data - vis_params["vmin"]) / (vis_params["vmax"] - vis_params["vmin"])
                    norm_data = np.clip(norm_data, 0, 1)
                    
                    # Mask NaN values
                    mask = np.isnan(data)
                    norm_data = np.ma.masked_array(norm_data, mask=mask)
                    
                    # Store the data and parameters
                    band_data.append(norm_data)
                    vis_params_list.append(vis_params)
                    band_names_found.append(description)
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Band '{band_name}' not found in the input image")
        
        # Determine figure aspect ratio based on stack orientation
        if horizontal:
            figsize = (15, 8)
        else:
            figsize = (10, 12)
        
        # Set up the 3D figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each band as a layer in the stack
        for i, (data, vis_params, band_name) in enumerate(zip(band_data, vis_params_list, band_names_found)):
            # Set position for this layer based on orientation
            if horizontal:
                # Position along x-axis for horizontal stacking
                position = i * spacing
                # Shift X coordinates for each layer
                surface_x = X + position
                surface_y = Y
                surface_z = np.zeros_like(data)  # Flat surface at z=0
            else:
                # Position along z-axis for vertical stacking
                position = i * spacing
                surface_x = X
                surface_y = Y
                surface_z = position * np.ones_like(data)
            
            # Create custom colormap for categorical data if needed
            cmap = vis_params["cmap"]
            if not vis_params["continuous"] and "colors" in vis_params:
                cmap = ListedColormap(vis_params["colors"])
            
            # Plot the surface
            stride = max(1, data.shape[0] // 50)  # Reduce resolution for performance
            surf = ax.plot_surface(
                surface_x, surface_y, surface_z,
                rstride=stride, 
                cstride=stride,
                facecolors=plt.cm.get_cmap(cmap)(data),
                shade=False,
                alpha=0.9
            )
        
        # Configure the view and labels based on orientation
        if horizontal:
            # For horizontal stacking
            ax.set_xlabel('Longitude + Band Offset')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Value')
            
            # Set view angle for horizontal stack
            ax.view_init(elev=30, azim=-60)
            
            # Hide z ticks as they're not meaningful
            ax.set_zticks([])
        else:
            # For vertical stacking
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            # Hide z-axis for vertical stack
            ax.set_zticks([])
            ax.set_zlim(0, (len(band_data) - 1) * spacing + spacing)
            
            # Set view angle for vertical stack
            ax.view_init(elev=25, azim=-60)
        
        # Hide all panes except the bottom (xy) pane
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Add a title with band names
        orientation_text = "Horizontal" if horizontal else "Vertical"
        plt.title(f"{orientation_text} Stack of Bands: {', '.join(band_names_found)}")
        
        # Hide grid
        ax.grid(False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create a filename for the plot
        stack_type = "horizontal" if horizontal else "vertical"
        if output_filename:
            filename = f"{output_filename}.png"
        else:
            filename = f"image_stack_3d_{stack_type}.png"
            
        fig_path = output_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"3D {stack_type} stack visualization saved to {fig_path}")

    print(f"All visualizations saved to {output_dir}") 
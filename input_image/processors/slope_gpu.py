"""
DEM processing functions with GPU optimization using OpenCL for AMD GPUs.
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import time

from ..config import UNIVERSAL_DTYPE, UNIVERSAL_NODATA, GEOTIFF_EXT, UNIVERSAL_CRS

# GPU/OpenCL support
OPENCL_AVAILABLE = False
ROCM_AVAILABLE = False

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    pass

try:
    # This is a simple check to see if ROCm Python packages are available
    # We could use rocm-smi via subprocess as well, but this is quicker for checking
    import torch
    if torch.cuda.is_available() and torch.version.hip is not None:
        ROCM_AVAILABLE = True
except ImportError:
    pass

# OpenCL kernel for Horn's method slope calculation
SLOPE_KERNEL_CODE = """
__kernel void calculate_slope(
    __global const float *dem,
    __global float *slope,
    const int width,
    const int height,
    const float cell_size_x,
    const float cell_size_y)
{
    // Get global position
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int idx = y * width + x;
    
    // Skip border pixels
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        slope[idx] = NAN;
        return;
    }
    
    // Get neighboring elevations using Horn's method
    float z1 = dem[(y-1) * width + (x-1)];
    float z2 = dem[(y-1) * width + x];
    float z3 = dem[(y-1) * width + (x+1)];
    float z4 = dem[y * width + (x-1)];
    float z6 = dem[y * width + (x+1)];
    float z7 = dem[(y+1) * width + (x-1)];
    float z8 = dem[(y+1) * width + x];
    float z9 = dem[(y+1) * width + (x+1)];
    
    // Check if any neighboring values are NaN
    if (isnan(z1) || isnan(z2) || isnan(z3) || isnan(z4) || isnan(z6) || isnan(z7) || isnan(z8) || isnan(z9)) {
        slope[idx] = NAN;
        return;
    }
    
    // Calculate gradients using Horn's method
    float dzdx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8.0f * cell_size_x);
    float dzdy = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8.0f * cell_size_y);
    
    // Calculate slope
    float slope_value = sqrt(dzdx*dzdx + dzdy*dzdy);
    
    // Convert to degrees
    slope[idx] = atan(slope_value) * (180.0f / M_PI);
}
"""


def compute_slope_opencl(dem_array, cell_size_x, cell_size_y):
    """
    Calculate slope from DEM using OpenCL on GPU.
    
    Args:
        dem_array: Input DEM as numpy array
        cell_size_x: Cell width in map units
        cell_size_y: Cell height in map units
    
    Returns:
        Slope array in degrees
    """
    # Get shape
    height, width = dem_array.shape
    
    # Initialize OpenCL
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print("Warning: No OpenCL platforms found")
            return None
        
        # Prefer AMD platform if available
        amd_platform = None
        for platform in platforms:
            if 'amd' in platform.vendor.lower():
                amd_platform = platform
                break
        
        # If no AMD platform, use the first available
        platform = amd_platform if amd_platform else platforms[0]
        
        # Get device (prefer GPU)
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            devices = platform.get_devices()
            if not devices:
                print("Warning: No OpenCL devices found")
                return None
        
        device = devices[0]
        print(f"Using OpenCL device: {device.name}")
        
        # Create context and queue
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        
        # Ensure input array is float32
        dem_array_f32 = dem_array.astype(np.float32)
        
        # Create buffers
        dem_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                            hostbuf=dem_array_f32)
        slope_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 
                             dem_array_f32.nbytes)
        
        # Build program
        program = cl.Program(ctx, SLOPE_KERNEL_CODE).build()
        
        # Execute kernel
        program.calculate_slope(queue, (width, height), None,
                              dem_buf, slope_buf, np.int32(width), np.int32(height),
                              np.float32(cell_size_x), np.float32(cell_size_y))
        
        # Read result
        slope_array = np.empty_like(dem_array_f32)
        cl.enqueue_copy(queue, slope_array, slope_buf)
        
        return slope_array
        
    except Exception as e:
        print(f"OpenCL error: {str(e)}")
        return None


def compute_slope_rocm(dem_array, cell_size_x, cell_size_y):
    """
    Calculate slope from DEM using ROCm (via PyTorch).
    Currently a placeholder that could be implemented if needed.
    
    Args:
        dem_array: Input DEM as numpy array
        cell_size_x: Cell width in map units
        cell_size_y: Cell height in map units
    
    Returns:
        Slope array in degrees or None if not implemented/available
    """
    try:
        import torch
        
        if not torch.cuda.is_available() or torch.version.hip is None:
            print("ROCm/HIP not available in PyTorch")
            return None
            
        # This is a placeholder for actual ROCm implementation
        # A full implementation would use custom HIP kernels or PyTorch operations
        print("ROCm implementation not currently available")
        return None
        
    except ImportError:
        print("PyTorch with ROCm support not available")
        return None


def compute_slope(dem_path, bands_dir, force_cpu=False):
    """Compute slope from DEM using GPU acceleration if available,
    falling back to CPU if needed
    
    Args:
        dem_path (str or Path): Path to the DEM file
        bands_dir (str or Path): Directory for saving output
        force_cpu (bool): Whether to force CPU execution even if GPU is available
        
    Returns:
        Path: Path to the generated slope file
    """
    print("\n=== Starting compute_slope (GPU) ===")
    dem_path = Path(dem_path)
    bands_dir = Path(bands_dir)
    
    print(f"Loading DEM from: {dem_path}")
    
    # Get the DEM metadata and read the data
    with rasterio.open(dem_path) as src:
        dem_meta = src.meta.copy()
        transform = src.transform
        dem_array = src.read(1).astype(np.float32)
        
        # Calculate cell sizes
        cell_size_x = abs(transform[0])
        cell_size_y = abs(transform[4])
        
        print(f"DEM cell sizes - X: {cell_size_x}, Y: {cell_size_y}")
        print(f"DEM loaded, shape: {dem_array.shape}")
    
    # Try GPU implementations if not forced to use CPU
    slope_array = None
    
    if not force_cpu:
        if OPENCL_AVAILABLE:
            print("Attempting OpenCL implementation...")
            start_time = time.time()
            slope_array = compute_slope_opencl(dem_array, cell_size_x, cell_size_y)
            if slope_array is not None:
                print(f"OpenCL computation completed in {time.time() - start_time:.2f} seconds")
        
        if slope_array is None and ROCM_AVAILABLE:
            print("Attempting ROCm implementation...")
            start_time = time.time()
            slope_array = compute_slope_rocm(dem_array, cell_size_x, cell_size_y)
            if slope_array is not None:
                print(f"ROCm computation completed in {time.time() - start_time:.2f} seconds")
    
    # Fall back to CPU implementation if GPU failed or was forced off
    if slope_array is None:
        print("Falling back to CPU implementation...")
        
        # Import CPU implementation
        from .slope_optimized import calculate_slope_numba
        
        start_time = time.time()
        cell_size = (cell_size_x, cell_size_y)
        slope_array = calculate_slope_numba(dem_array, cell_size, neighbors=8, units="degrees")
        print(f"CPU computation completed in {time.time() - start_time:.2f} seconds")
    
    # Define output path
    slope_path = bands_dir / f"slope{GEOTIFF_EXT}"
    print(f"Writing slope to: {slope_path}")
    
    # Update metadata for slope raster
    dem_meta.update({
        'dtype': UNIVERSAL_DTYPE,
        'count': 1,
        'nodata': UNIVERSAL_NODATA
    })
    
    # Write the slope raster
    with rasterio.open(slope_path, 'w', **dem_meta) as dst:
        dst.write(slope_array[np.newaxis, :, :].astype(dem_meta['dtype']))
        dst.set_band_description(1, "Terrain slope (degrees)")
    
    # Print statistics
    valid_slope = slope_array[~np.isnan(slope_array)]
    if len(valid_slope) > 0:
        print(f"Slope statistics - min: {np.min(valid_slope):.2f}°, max: {np.max(valid_slope):.2f}°, mean: {np.mean(valid_slope):.2f}°")
    else:
        print("Warning: No valid slope values found")
    
    print("=== Completed compute_slope (GPU) ===\n")
    
    return slope_path 
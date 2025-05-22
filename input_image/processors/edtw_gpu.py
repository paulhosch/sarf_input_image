"""
EDTW processing functions with GPU optimization using OpenCL for AMD GPUs.
"""

import os
import numpy as np
import rasterio
from geopy.distance import geodesic
import time
from pathlib import Path

from ..config import UNIVERSAL_DTYPE, UNIVERSAL_NODATA
from ..utils.print_info import log_export_info

# GPU/OpenCL support
OPENCL_AVAILABLE = False
ROCM_AVAILABLE = False

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    pass

try:
    # Check for ROCm support
    import torch
    if torch.cuda.is_available() and torch.version.hip is not None:
        ROCM_AVAILABLE = True
except ImportError:
    pass

# OpenCL kernel for Euclidean distance transform
EDTW_KERNEL_CODE = """
__kernel void euclidean_distance(
    __global const unsigned char *binary_mask,
    __global float *distance,
    const int width,
    const int height,
    const float scale_x,
    const float scale_y)
{
    // Get global position
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int idx = y * width + x;
    
    // Skip if this is already a feature pixel (water)
    if (binary_mask[idx] == 0) {
        distance[idx] = 0.0f;
        return;
    }
    
    // Search for nearest feature pixel
    float min_dist = 1.0e10f; // Very large initial value
    
    // Define reasonable search radius based on image size
    // This is a performance optimization - there's a trade-off here
    // between search radius and accuracy for very large images
    int search_radius = min(min(width, height) / 2, 100);
    
    int start_x = max(0, x - search_radius);
    int end_x = min(width, x + search_radius + 1);
    int start_y = max(0, y - search_radius);
    int end_y = min(height, y + search_radius + 1);
    
    for (int ny = start_y; ny < end_y; ny++) {
        for (int nx = start_x; nx < end_x; nx++) {
            int n_idx = ny * width + nx;
            if (binary_mask[n_idx] == 0) {
                // Calculate squared distance with proper scaling
                float dx = (x - nx) * scale_x;
                float dy = (y - ny) * scale_y;
                float dist = sqrt(dx*dx + dy*dy);
                
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }
    }
    
    distance[idx] = min_dist;
}
"""


def get_cell_size(water_path):
    """Get pixel size in meters based on geographic coordinates"""
    with rasterio.open(water_path) as src:
        transform = src.transform
        crs = src.crs
        
        # Get pixel width and height in degrees
        pixel_width_deg = transform.a
        pixel_height_deg = -transform.e

        # Center of the raster to estimate latitude
        center_lat = src.bounds.top - (src.height // 2) * pixel_height_deg
        center_lon = src.bounds.left + (src.width // 2) * pixel_width_deg

        # Approximate meters per pixel using geodesic distance
        pixel_width_m = geodesic(
            (center_lat, center_lon),
            (center_lat, center_lon + pixel_width_deg)
        ).meters

        pixel_height_m = geodesic(
            (center_lat, center_lon),
            (center_lat + pixel_height_deg, center_lon)
        ).meters

    print(f"Pixel size: {pixel_width_m:.2f} m x {pixel_height_m:.2f} m")
    return pixel_width_m, pixel_height_m


def compute_edtw_opencl(binary_raster, pixel_width_m, pixel_height_m):
    """
    Compute Euclidean distance transform using OpenCL on GPU.
    
    Args:
        binary_raster: Binary water raster (0 for water, 1 for non-water)
        pixel_width_m: Pixel width in meters
        pixel_height_m: Pixel height in meters
        
    Returns:
        Distance array or None if computation failed
    """
    # Get shape
    height, width = binary_raster.shape
    
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
        
        # Ensure input array is unsigned char
        binary_raster_u8 = binary_raster.astype(np.uint8)
        
        # Create output array
        distance_array = np.zeros(binary_raster.shape, dtype=np.float32)
        
        # Create buffers
        binary_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                             hostbuf=binary_raster_u8)
        dist_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 
                            distance_array.nbytes)
        
        # Build program
        program = cl.Program(ctx, EDTW_KERNEL_CODE).build()
        
        # Execute kernel
        program.euclidean_distance(queue, (width, height), None,
                                 binary_buf, dist_buf, np.int32(width), np.int32(height),
                                 np.float32(pixel_width_m), np.float32(pixel_height_m))
        
        # Read result
        cl.enqueue_copy(queue, distance_array, dist_buf)
        
        return distance_array
        
    except Exception as e:
        print(f"OpenCL error: {str(e)}")
        return None


def compute_edtw_rocm(binary_raster, pixel_width_m, pixel_height_m):
    """
    Compute Euclidean distance transform using ROCm (via PyTorch).
    Currently a placeholder that could be implemented if needed.
    
    Args:
        binary_raster: Binary water raster (0 for water, 1 for non-water)
        pixel_width_m: Pixel width in meters
        pixel_height_m: Pixel height in meters
        
    Returns:
        Distance array or None if not implemented/available
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


def compute_euclidean_distance(water_path, output_dir, force_cpu=False):
    """
    Compute Euclidean distance to nearest non-zero pixel using
    GPU acceleration if available, falling back to CPU if needed.
    
    Args:
        water_path (str): Path to binary water raster
        output_dir (str): Directory for saving results
        force_cpu (bool): Whether to force CPU execution even if GPU is available
        
    Returns:
        numpy.ndarray: Distance transform
    """
    print("\n=== Starting compute_euclidean_distance (GPU) ===")
    water_path = Path(water_path)
    output_dir = Path(output_dir)
    
    # Read binary water raster
    with rasterio.open(water_path) as src:
        binary_raster = src.read(1)
        meta = src.meta.copy()
    
    # Get cell size in meters
    pixel_width_m, pixel_height_m = get_cell_size(water_path)

    water_pixels = np.sum(binary_raster == 0)
    total_pixels = binary_raster.size
    print(f"Computing distance from {water_pixels} water pixels ({water_pixels/total_pixels*100:.2f}% of raster)")
    
    # Try GPU implementations if not forced to use CPU
    distance = None
    
    if not force_cpu:
        if OPENCL_AVAILABLE:
            print("Attempting OpenCL implementation...")
            start_time = time.time()
            distance = compute_edtw_opencl(binary_raster, pixel_width_m, pixel_height_m)
            if distance is not None:
                print(f"OpenCL computation completed in {time.time() - start_time:.2f} seconds")
        
        if distance is None and ROCM_AVAILABLE:
            print("Attempting ROCm implementation...")
            start_time = time.time()
            distance = compute_edtw_rocm(binary_raster, pixel_width_m, pixel_height_m)
            if distance is not None:
                print(f"ROCm computation completed in {time.time() - start_time:.2f} seconds")
    
    # Fall back to CPU implementation if GPU failed or was forced off
    if distance is None:
        print("Falling back to CPU implementation...")
        
        # Check if water pixels are too sparse - if so, use the standard distance transform
        water_percentage = water_pixels / total_pixels
        
        if water_percentage < 0.01 or binary_raster.size > 25000000:  # If < 1% water or large raster
            print("Using scipy's distance_transform_edt due to sparse water or large raster...")
            from scipy.ndimage import distance_transform_edt
            start_time = time.time()
            distance = distance_transform_edt(binary_raster != 0, sampling=(pixel_height_m, pixel_width_m))
            print(f"SciPy computation completed in {time.time() - start_time:.2f} seconds")
        else:
            print("Using Numba-optimized distance transform...")
            # Import CPU implementation
            from .edtw_optimized import optimized_edt
            
            start_time = time.time()
            distance = optimized_edt(binary_raster, sampling=(pixel_height_m, pixel_width_m))
            print(f"Numba computation completed in {time.time() - start_time:.2f} seconds")
    
    # Save distance transform to intermediate directory if provided
    edtw_path = output_dir / 'edtw.tif'
    print(f"Writing EDTW to: {edtw_path}")
    
    meta.update({
        'dtype': UNIVERSAL_DTYPE,
        'nodata': UNIVERSAL_NODATA
    })
    
    with rasterio.open(edtw_path, 'w', **meta) as dst:
        dst.write(distance[np.newaxis, :, :])
        
    # Print information about the resulting GeoTIFF
    with rasterio.open(edtw_path) as src:
        log_export_info("EDTW", src, is_ee=False)
    
    print(f"Distance min: {distance.min()}, max: {distance.max()}, mean: {distance.mean():.2f}")
    print("=== Completed compute_euclidean_distance (GPU) ===\n")
    return distance 
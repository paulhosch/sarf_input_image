import os
import numpy as np
import rasterio
from input_image.utils import logger, log_execution

@log_execution
def compute_sar_derivatives(output_dir):
    """Compute SAR derivatives from individual VV and VH bands
    
    Args:
        output_dir (str): Directory with output files
        
    Returns:
        dict: Paths to all generated SAR products
    """
    logger.info("=== Starting compute_sar_derivatives ===")
    
    # Paths to input files
    vv_pre_path = os.path.join(output_dir, 'VV_pre.tif')
    vh_pre_path = os.path.join(output_dir, 'VH_pre.tif')
    vv_post_path = os.path.join(output_dir, 'VV_post.tif')
    vh_post_path = os.path.join(output_dir, 'VH_post.tif')
    
    # Check if all required files exist
    required_files = [vv_pre_path, vh_pre_path, vv_post_path, vh_post_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load data using rasterio
    logger.info("Loading SAR data from TIF files...")
    with rasterio.open(vv_pre_path) as src:
        vv_pre = src.read(1)
        profile = src.profile.copy()
    
    with rasterio.open(vh_pre_path) as src:
        vh_pre = src.read(1)
    
    with rasterio.open(vv_post_path) as src:
        vv_post = src.read(1)
    
    with rasterio.open(vh_post_path) as src:
        vh_post = src.read(1)
    
    # Compute ratios
    logger.info("Computing VV/VH ratio for pre-event image...")
    pre_ratio = np.divide(vv_pre, vh_pre, out=np.zeros_like(vv_pre), where=vh_pre!=0)
    
    logger.info("Computing VV/VH ratio for post-event image...")
    post_ratio = np.divide(vv_post, vh_post, out=np.zeros_like(vv_post), where=vh_post!=0)
    
    # Compute change images
    logger.info("Computing change detection bands...")
    vv_change = np.divide(vv_post, vv_pre, out=np.zeros_like(vv_post), where=vv_pre!=0)
    vh_change = np.divide(vh_post, vh_pre, out=np.zeros_like(vh_post), where=vh_pre!=0)
    ratio_change = np.divide(post_ratio, pre_ratio, out=np.zeros_like(post_ratio), where=pre_ratio!=0)
    
    # Save computed derivatives as individual TIFs
    output_files = {}
    
    # Define output paths
    pre_ratio_path = os.path.join(output_dir, 'VV_VH_ratio_pre.tif')
    post_ratio_path = os.path.join(output_dir, 'VV_VH_ratio_post.tif')
    vv_change_path = os.path.join(output_dir, 'VV_change.tif')
    vh_change_path = os.path.join(output_dir, 'VH_change.tif')
    ratio_change_path = os.path.join(output_dir, 'VV_VH_ratio_change.tif')
    
    # Write files
    logger.info("Saving derivative bands...")
    
    # Save pre_ratio
    with rasterio.open(pre_ratio_path, 'w', **profile) as dst:
        dst.write(pre_ratio, 1)
    output_files['pre_ratio'] = pre_ratio_path
    
    # Save post_ratio
    with rasterio.open(post_ratio_path, 'w', **profile) as dst:
        dst.write(post_ratio, 1)
    output_files['post_ratio'] = post_ratio_path
    
    # Save vv_change
    with rasterio.open(vv_change_path, 'w', **profile) as dst:
        dst.write(vv_change, 1)
    output_files['vv_change'] = vv_change_path
    
    # Save vh_change
    with rasterio.open(vh_change_path, 'w', **profile) as dst:
        dst.write(vh_change, 1)
    output_files['vh_change'] = vh_change_path
    
    # Save ratio_change
    with rasterio.open(ratio_change_path, 'w', **profile) as dst:
        dst.write(ratio_change, 1)
    output_files['ratio_change'] = ratio_change_path
    
    logger.info("All SAR derivatives computed and saved")
    logger.info("=== Completed compute_sar_derivatives ===")
    
    return output_files
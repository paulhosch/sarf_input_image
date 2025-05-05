from input_image.config import UNIVERSAL_CRS, UNIVERSAL_DTYPE, UNIVERSAL_NODATA
from .logger import logger

# Export info logger format
def log_export_info(name, data, is_ee=False):
    """Log standardized information about exported images
    
    Args:
        name (str): Name of the dataset
        data: The image data (ee.Image or numpy array with metadata)
        is_ee (bool): Whether this is an Earth Engine image
    """
    if is_ee:
        try:
            logger.info(f"--- {name} Export Information (Earth Engine) ---")
            logger.info("EE Image Metadata:")
            metadata = data.getInfo()
            for key, value in metadata.items():
                logger.info(f"{key}: {value}")
            logger.info(f"Universal Export Settings:")
            logger.info(f"Target CRS: {UNIVERSAL_CRS}")
            logger.info(f"Target dtype: {UNIVERSAL_DTYPE}")
            logger.info(f"Target nodata: {UNIVERSAL_NODATA}")
        except Exception as e:
            logger.warning(f"Unable to get full EE image info: {e}")
    else:
        try:
            logger.info(f"--- {name} Export Information (GeoTIFF) ---")
            logger.info(f"CRS: {data.crs}")
            res_x = abs(data.transform[0])
            res_y = abs(data.transform[4])
            logger.info(f"Resolution: {res_x:.7f}° x {res_y:.7f}° (~{res_x*111319:.1f}m x {res_y*111319:.1f}m)")
            logger.info(f"Data type: {data.dtypes[0]}")
            logger.info(f"NoData value: {data.nodata}")
            logger.info(f"Size: {data.width} x {data.height} pixels")
        except Exception as e:
            logger.warning(f"Unable to get full GeoTIFF info: {e}")
    logger.info("-------------------------------------------") 
"""
Earth Engine initialization functions.
"""

import ee

def initialize_ee():
    """Initialize Earth Engine with high-volume API endpoint
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    print("\n=== Starting Earth Engine initialization ===")
    
    try:
        # Initialize with high-volume API endpoint
        ee.Initialize() #opt_url='https://earthengine-highvolume.googleapis.com'
        print("Initialized Earth Engine with high-volume API endpoint")
        
        # Try to get info about something simple to verify connection
        info = ee.Image('USGS/SRTMGL1_003').getInfo()
        print("Earth Engine connection verified")
        print("=== Completed Earth Engine initialization ===\n")
         
            
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("Please authenticate with Earth Engine using 'earthengine authenticate' or specify the project id")
        print("=== Failed Earth Engine initialization ===\n")

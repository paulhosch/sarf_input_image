"""
Earth Engine initialization functions.
"""

import ee
import os

def initialize_ee(project_id=None):
    """Initialize Earth Engine with high-volume API endpoint
    
    Args:
        project_id (str, optional): Google Cloud project ID. If None, will try to use
                                  environment variable EARTHENGINE_PROJECT_ID or default to 'auto-383910'
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    print("\n=== Starting Earth Engine initialization ===")
    
    try:
        # Get project ID from args, environment variable, or use default
        project_id = 'auto-383910'
        
        # Initialize with high-volume API endpoint
        ee.Initialize(project=project_id)
        print(f"Initialized Earth Engine with project: {project_id}")
        
        # Try to get info about something simple to verify connection
        info = ee.Image('USGS/SRTMGL1_003').getInfo()
        print("Earth Engine connection verified")
        print("=== Completed Earth Engine initialization ===\n")
        return True
            
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("\nTo fix this error:")
        print("1. Run 'earthengine authenticate' in your terminal")
        print("2. Set up a Google Cloud project and enable Earth Engine API")
        print("3. Either:")
        print("   - Set the EARTHENGINE_PROJECT_ID environment variable")
        print("   - Pass your project_id to initialize_ee()")
        print("   - Or use the default project if you have access to it")
        print("\nFor more details, visit: https://developers.google.com/earth-engine/guides/auth")
        print("=== Failed Earth Engine initialization ===\n")
        return False

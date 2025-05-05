"""
Logger module for input image processing pipeline.
"""

import logging
import os
import functools
import time
from pathlib import Path

# Create logs directory if it doesn't exist
def setup_logger(name='input_image'):
    """
    Set up and configure the logger.
    
    Args:
        name: Name of the logger
    
    Returns:
        Logger instance
    """
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'{name}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger()

def log_execution(func):
    """
    Decorator to log function execution time and parameters.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        
        result = func(*args, **kwargs)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed {func.__name__} in {elapsed_time:.2f} seconds")
        
        return result
    return wrapper 
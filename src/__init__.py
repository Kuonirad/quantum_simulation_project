"""
Quantum Simulation Project
=========================

A comprehensive quantum simulation framework with advanced visualization capabilities.

This package provides a robust implementation of quantum mechanical simulations
with integrated GUI visualization, testing frameworks, and scientific computation tools.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Package metadata
__version__ = "1.0.0"
__author__ = "Kuonirad"
__license__ = "MIT"

# Environment configuration
ENV_CONFIG: Dict[str, Any] = {
    "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "offscreen"),
    "QT_OPENGL": os.environ.get("QT_OPENGL", "software"),
    "PYTHONPATH": str(Path(__file__).parent.absolute()),
}


def setup_environment() -> None:
    """Configure the environment for both GUI and headless operation."""
    try:
        # Set critical environment variables
        for key, value in ENV_CONFIG.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.info(f"Set environment variable {key}={value}")

        # Add package root to Python path
        if ENV_CONFIG["PYTHONPATH"] not in sys.path:
            sys.path.insert(0, ENV_CONFIG["PYTHONPATH"])
            logger.info(f"Added {ENV_CONFIG['PYTHONPATH']} to Python path")

    except Exception as e:
        logger.error(f"Failed to setup environment: {str(e)}")
        raise


def verify_dependencies() -> bool:
    """Verify all required dependencies are installed and compatible."""
    try:
        import numpy
        import OpenGL
        import PyQt5
        import pyqtgraph
        import scipy

        logger.info("All core dependencies verified successfully")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False


def configure_gpu() -> Optional[str]:
    """Configure GPU settings for OpenGL rendering."""
    try:
        import OpenGL.GL as gl

        # Get OpenGL context information
        vendor = gl.glGetString(gl.GL_VENDOR).decode()
        renderer = gl.glGetString(gl.GL_RENDERER).decode()
        version = gl.glGetString(gl.GL_VERSION).decode()

        logger.info(f"OpenGL Vendor: {vendor}")
        logger.info(f"OpenGL Renderer: {renderer}")
        logger.info(f"OpenGL Version: {version}")

        return renderer
    except Exception as e:
        logger.warning(f"GPU configuration failed: {str(e)}")
        return None


# Initialize package
setup_environment()
if not verify_dependencies():
    logger.warning("Some dependencies are missing. Please install required packages.")

# Configure GPU if available
gpu_info = configure_gpu()
if gpu_info:
    logger.info(f"GPU configured successfully: {gpu_info}")
else:
    logger.info("Running in CPU-only mode")

# Export public interface
__all__ = ["setup_environment", "verify_dependencies", "configure_gpu", "__version__", "__author__", "__license__"]

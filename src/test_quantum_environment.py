# -*- coding: utf-8 -*-
"""
Test script for verifying quantum environment setup.
"""

import importlib
import sys


def check_package(package_name):
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is installed")
        return True
    except ImportError:
        print(f"{package_name} is not installed")
        return False


def check_environment():
    print(f"Python version: {sys.version}")

    packages = ["PyQt5", "pyqtgraph", "OpenGL"]
    all_installed = all(check_package(pkg) for pkg in packages)

    if all_installed:
        try:
            from PyQt5.QtWidgets import QApplication

            QApplication([])  # Initialize QApplication
            print("QApplication initialized successfully")
        except Exception as e:
            print(f"Error initializing QApplication: {e}")

        try:
            import pyqtgraph.opengl as gl

            gl.GLViewWidget()  # Initialize GLViewWidget
            print("GLViewWidget initialized successfully")
        except Exception as e:
            print(f"Error initializing GLViewWidget: {e}")
    else:
        print("Not all required packages are installed. Please install missing packages.")


if __name__ == "__main__":
    check_environment()

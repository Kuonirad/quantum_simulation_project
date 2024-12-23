# -*- coding: utf-8 -*-
"""
Verification script for quantum GUI components.
"""

import sys

from PyQt5.QtWidgets import QApplication, QPushButton, QSlider

from quantum_gui import QuantumSimulationGUI


def verify_quantum_gui():
    QApplication(sys.argv)  # Initialize QApplication
    gui = QuantumSimulationGUI()

    print("Interactivity check:")
    print(f"Number of sliders: {len(gui.findChildren(QSlider))}")
    print(f"Number of buttons: {len(gui.findChildren(QPushButton))}")
    print(f"Number of tabs: {gui.tabs.count()}")

    print("\nLuxCore integration check:")
    print(f'LuxCore thread exists: {hasattr(gui, "luxcore_thread")}')
    print(f'LuxCore render button exists: {hasattr(gui, "luxcore_render_button")}')

    print("\nAdvanced experimentation features check:")
    print(f'Perturbation feature exists: {hasattr(gui, "apply_perturbation")}')
    print(f'Entanglement analysis exists: {hasattr(gui, "analyze_entanglement")}')
    print(f'Quantum error correction exists: {hasattr(gui, "quantum_error_correction")}')

    print("\nVisualization methods check:")
    viz_methods = [
        "plot_wavefunction",
        "plot_probability_density",
        "plot_phase",
        "plot_momentum_space",
        "plot_isosurface",
        "plot_streamplot",
        "plot_poincare_section",
    ]
    for method in viz_methods:
        print(f"{method} exists: {hasattr(gui, method)}")


if __name__ == "__main__":
    verify_quantum_gui()

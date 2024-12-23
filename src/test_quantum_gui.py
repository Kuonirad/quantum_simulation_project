# -*- coding: utf-8 -*-
"""
Test script for quantum GUI functionality.
"""

import sys

from PyQt5.QtWidgets import QApplication, QWidget

from quantum_gui import QuantumSimulationGUI


def test_quantum_gui():
    QApplication(sys.argv)  # Initialize QApplication
    gui = QuantumSimulationGUI()

    print("GUI initialized successfully")
    print("Number of tabs:", gui.tabs.count())
    print("Tab names:", [gui.tabs.tabText(i) for i in range(gui.tabs.count())])
    print(
        "Number of sliders in wavefunction tab:",
        len([w for w in gui.tabs.widget(0).findChildren(QWidget) if "slider" in w.objectName().lower()]),
    )
    print("3D scatter plot initialized:", hasattr(gui, "scatter") and gui.scatter is not None)
    print("Blender visualization button exists:", hasattr(gui, "blender_button") and gui.blender_button is not None)

    # Test interactivity
    print("\nTesting interactivity:")
    if hasattr(gui, "run_button"):
        print("Run button exists")
        gui.run_button.click()
        print("Run button clicked")

    if hasattr(gui, "reset_button"):
        print("Reset button exists")
        gui.reset_button.click()
        print("Reset button clicked")

    if hasattr(gui, "viz_type") and gui.viz_type.count() > 0:
        print("Visualization type selector exists")
        gui.viz_type.setCurrentIndex(1)
        print("Changed visualization type")

    # Test advanced parameters
    if hasattr(gui, "magnetic_field"):
        print("Magnetic field slider exists")
        gui.magnetic_field.setValue(5.0)
        print("Set magnetic field to 5.0")

    if hasattr(gui, "perturbation_strength"):
        print("Perturbation strength slider exists")
        gui.perturbation_strength.setValue(0.5)
        print("Set perturbation strength to 0.5")

    print("\nGUI testing completed")


if __name__ == "__main__":
    test_quantum_gui()

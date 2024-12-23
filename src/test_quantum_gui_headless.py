import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

matplotlib.use("Agg")  # Use non-interactive backend


class HeadlessQuantumGUI:
    def __init__(self):
        self.wavefunction = None
        self.current_step = 0
        self.parameter_history = []
        self.n_steps = 100
        self.dt = 0.01
        self.l = 1
        self.m = 0
        self.energy = 1.0
        self.potential = 0.5
        self.spin = 0.5
        self.magnetic_field = 1.0
        self.electric_field = 0.5
        self.temperature = 300
        print("HeadlessQuantumGUI initialized")

    def calculate_wavefunction(self):
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        self.wavefunction = np.exp(-(R**2) / 2) * np.cos(self.m * theta)
        self.current_step += 1

    def visualize_wavefunction(self):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        x = y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.abs(self.wavefunction) ** 2
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Probability Density")
        ax.set_title(f"3D Wavefunction Visualization (Step {self.current_step})")
        plt.colorbar(surf, label="Probability Density")
        filename = f"wavefunction_3d_step_{self.current_step}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Wavefunction visualization saved: {filename}")

    def update_simulation(self):
        self.calculate_wavefunction()
        self.visualize_wavefunction()
        self.parameter_history.append(
            {
                "step": self.current_step,
                "energy": self.energy,
                "potential": self.potential,
                "magnetic_field": self.magnetic_field,
                "electric_field": self.electric_field,
                "temperature": self.temperature,
            }
        )
        print(f"Simulation step {self.current_step} completed")

    def run_simulation(self):
        print("Starting simulation")
        for _ in range(self.n_steps):
            self.update_simulation()
        print("Simulation completed")
        self.visualize_parameter_history()

    def visualize_parameter_history(self):
        plt.figure(figsize=(12, 10))
        steps = [p["step"] for p in self.parameter_history]
        for param in ["energy", "potential", "magnetic_field", "electric_field", "temperature"]:
            values = [p[param] for p in self.parameter_history]
            plt.plot(steps, values, label=param.capitalize())
        plt.xlabel("Simulation Step")
        plt.ylabel("Parameter Value")
        plt.title("Parameter Evolution During Simulation")
        plt.legend()
        plt.savefig("parameter_history.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Parameter history visualization saved: parameter_history.png")


def run_headless_simulation():
    print("Starting headless quantum simulation")
    QApplication(sys.argv)  # Initialize QApplication
    gui = HeadlessQuantumGUI()
    gui.run_simulation()
    print("Headless quantum simulation completed")


if __name__ == "__main__":
    run_headless_simulation()

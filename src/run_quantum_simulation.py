# -*- coding: utf-8 -*-
"""
Main script for running quantum simulations and saving results.
"""

import json
import logging

import matplotlib.pyplot as plt
import numpy as np

from quantum_simulation import run_russell_simulation, save_quantum_state_for_blender

# Configure logging
logging.basicConfig(
    filename="quantum_simulation.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Add console handler for INFO level logs
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


def run_simulation_and_save_results():
    logging.info("Starting quantum simulation")

    # Set simulation parameters
    n_steps = 100
    dt = 0.01
    angular_momentum = 2  # Angular momentum quantum number
    m = 1
    energy = 10
    potential = 5
    spin = 0.5
    magnetic_field = 1.0
    electric_field = 0.5
    temperature = 300
    perturbation_strength = 0.1
    quantum_noise = 0.01
    decoherence_rate = 0.001

    # Run simulation
    results = run_russell_simulation(
        n_steps,
        dt,
        angular_momentum,
        m,
        energy,
        potential,
        spin,
        magnetic_field,
        electric_field,
        temperature,
        perturbation_strength,
        quantum_noise,
        decoherence_rate,
    )

    # Save results
    np.save("psi_values.npy", results["psi_values"])
    np.save("entanglement_entropy.npy", results["entanglement_entropy"])
    np.save("spin_orbit_energy.npy", results["spin_orbit_energy"])

    # Generate plots
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(results["psi_values"][-1]) ** 2)
    plt.title("Final Wavefunction Probability Density")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.plot(results["entanglement_entropy"])
    plt.title("Entanglement Entropy Evolution")
    plt.xlabel("Time Step")
    plt.ylabel("Entropy")

    plt.subplot(2, 2, 3)
    plt.plot(results["spin_orbit_energy"])
    plt.title("Spin-Orbit Energy Evolution")
    plt.xlabel("Time Step")
    plt.ylabel("Energy")

    plt.subplot(2, 2, 4)
    plt.imshow(results["aqal_evolution"][-1])
    plt.title("Final AQAL Quadrant Evolution")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("simulation_results.png")
    logging.info("Saved simulation results plot to simulation_results.png")

    # Save quantum state for Blender
    save_quantum_state_for_blender(results["psi_values"], results["X"], results["Y"], results["Z"])
    logging.info("Saved quantum state for Blender visualization")

    # Save simulation parameters
    params = {
        "n_steps": n_steps,
        "dt": dt,
        "angular_momentum": angular_momentum,
        "m": m,
        "energy": energy,
        "potential": potential,
        "spin": spin,
        "magnetic_field": magnetic_field,
        "electric_field": electric_field,
        "temperature": temperature,
        "perturbation_strength": perturbation_strength,
        "quantum_noise": quantum_noise,
        "decoherence_rate": decoherence_rate,
    }
    with open("simulation_parameters.json", "w") as f:
        json.dump(params, f, indent=2)
    logging.info("Saved simulation parameters to simulation_parameters.json")

    logging.info("Quantum simulation completed successfully")


if __name__ == "__main__":
    run_simulation_and_save_results()

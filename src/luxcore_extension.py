import logging

import pyluxcore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QPushButton


class LuxCoreThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, psi_values, X, Y, Z, samples, resolution):
        super().__init__()
        self.psi_values = psi_values
        self.X = X
        self.Y = Y
        self.Z = Z
        self.samples = samples
        self.resolution = resolution

    def run(self):
        logging.info("Starting LuxCore rendering process")
        try:
            config = pyluxcore.Properties()
            config.SetFromString(
                f"""
            renderengine.type = PATHCPU
            sampler.type = SOBOL
            film.width = {self.resolution[0]}
            film.height = {self.resolution[1]}
            """
            )

            scene = pyluxcore.Scene(config)

            for i in range(len(self.X)):
                pos = pyluxcore.Property()
                pos.Set(
                    pyluxcore.Property(
                        "transformation", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, self.X[i], self.Y[i], self.Z[i], 1]
                    )
                )
                pos.Set(pyluxcore.Property("material.type", ["matte"]))
                pos.Set(pyluxcore.Property("material.kd", [abs(self.psi_values[i]) ** 2, 0, 0]))
                scene.Parse(pos)

            render_config = scene.GetRenderConfig()
            session = pyluxcore.RenderSession(render_config)
            session.Start()

            for i in range(self.samples):
                session.UpdateStats()
                progress = int((i + 1) / self.samples * 100)
                self.progress_signal.emit(progress)
                session.WaitNewFrame()
                if i % 10 == 0:
                    session.GetFilm().Save()

            session.GetFilm().Save()
            logging.info("LuxCore rendering process completed")
            self.finished_signal.emit()
        except Exception as e:
            logging.error(f"Error in LuxCore rendering: {str(e)}")


def add_luxcore_functionality(gui):
    logging.info("Adding LuxCore functionality to GUI")
    try:
        # Add LuxCore rendering button
        luxcore_button = QPushButton("Render with LuxCore")
        luxcore_button.clicked.connect(gui.generate_luxcore_viz)
        gui.layout().addWidget(luxcore_button)

        # Add method to generate LuxCore visualization
        def generate_luxcore_viz(self):
            logging.info("Generating LuxCore visualization")
            try:
                if self.psi_values is not None:
                    resolution = (1920, 1080)  # Full HD resolution
                    self.luxcore_thread = LuxCoreThread(
                        self.psi_values, self.X, self.Y, self.Z, samples=1000, resolution=resolution
                    )
                    self.luxcore_thread.progress_signal.connect(self.update_progress)
                    self.luxcore_thread.finished_signal.connect(self.luxcore_finished)
                    self.luxcore_thread.start()
                    self.luxcore_button.setEnabled(False)
                    self.status_text.append("LuxCore visualization generation started...")
                else:
                    raise ValueError("No simulation data available for visualization")
            except Exception as e:
                logging.error(f"Error in generate_luxcore_viz: {str(e)}")
                self.status_text.append(f"Error: {str(e)}")

        # Add method to handle LuxCore rendering completion
        def luxcore_finished(self):
            self.luxcore_button.setEnabled(True)
            self.status_text.append("LuxCore visualization completed.")
            # TODO: Display the rendered image

        # Add methods to the GUI class
        gui.__class__.generate_luxcore_viz = generate_luxcore_viz
        gui.__class__.luxcore_finished = luxcore_finished

        logging.info("LuxCore functionality added successfully")
    except Exception as e:
        logging.error(f"Error adding LuxCore functionality: {str(e)}")


def enhance_interactivity(gui):
    logging.info("Enhancing GUI interactivity")
    try:
        # Add real-time parameter updates
        for slider in [
            gui.n_steps_slider,
            gui.dt_slider,
            gui.l_slider,
            gui.m_slider,
            gui.energy_slider,
            gui.potential_slider,
            gui.spin_slider,
            gui.magnetic_field,
            gui.electric_field,
            gui.temperature,
            gui.laser_intensity,
            gui.pressure,
            gui.perturbation_strength,
            gui.quantum_noise,
            gui.decoherence_rate,
            gui.entanglement_strength,
        ]:
            slider.valueChanged.connect(gui.on_parameter_change)

        # Add method for real-time parameter updates
        def on_parameter_change(self):
            # Update simulation parameters in real-time
            self.update_simulation()
            # Trigger visualization update
            self.update_visualization()

        gui.__class__.on_parameter_change = on_parameter_change

        logging.info("GUI interactivity enhanced successfully")
    except Exception as e:
        logging.error(f"Error enhancing GUI interactivity: {str(e)}")


# Main function to apply extensions


def apply_extensions(gui):
    add_luxcore_functionality(gui)
    enhance_interactivity(gui)
    logging.info("All extensions applied successfully")

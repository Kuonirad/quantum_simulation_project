"""
Comprehensive test suite for quantum visualization system.
"""

import logging
import os
import sys

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

# Configure environment for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_OPENGL"] = "software"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

from src.quantum_gui import QuantumGLWidget, QuantumSimulationGUI

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def app() -> QApplication:
    """Create QApplication instance with proper OpenGL configuration."""
    # Set application attributes before creating QApplication
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    # Configure OpenGL format
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def gui(app: QApplication, request: pytest.FixtureRequest) -> QuantumSimulationGUI:
    """Create QuantumSimulationGUI instance with OpenGL context."""
    gui = QuantumSimulationGUI()

    # Ensure OpenGL context is created and current
    if not gui.gl_widget.context().isValid():
        logger.error("Failed to create valid OpenGL context")
        pytest.skip("OpenGL context creation failed")

    gui.show()
    yield gui
    gui.close()


def test_initialization(gui: QuantumSimulationGUI) -> None:
    """Test proper initialization of GUI components."""
    assert gui is not None
    assert isinstance(gui.gl_widget, QuantumGLWidget)
    assert gui.gl_widget.gpu_accelerator is not None
    assert gui.gl_widget.state_buffer is not None


def test_shader_programs(gui: QuantumSimulationGUI) -> None:
    """Test shader program initialization and compilation."""
    gl_widget = gui.gl_widget

    # Skip test if OpenGL context is not valid
    if not gl_widget.context().isValid():
        pytest.skip("OpenGL context not valid")

    # Verify shader programs exist
    assert gl_widget.surface_program is not None
    assert gl_widget.volumetric_program is not None
    assert gl_widget.raytracing_program is not None

    # Verify shader compilation
    assert gl_widget.surface_program.isLinked()
    assert gl_widget.volumetric_program.isLinked()
    assert gl_widget.raytracing_program.isLinked()


def test_quantum_state_update(gui: QuantumSimulationGUI) -> None:
    """Test quantum state updates and visualization."""
    gl_widget = gui.gl_widget

    # Skip test if OpenGL context is not valid
    if not gl_widget.context().isValid():
        pytest.skip("OpenGL context not valid")

    # Create test quantum state
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    sigma = 2.0
    r2 = X**2 + Y**2
    psi = np.exp(-r2 / (4 * sigma**2)) * np.exp(1j * (X + Y))
    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2))

    # Update state and verify
    gl_widget.set_quantum_state(psi)
    current_state = gl_widget.get_current_state()
    assert current_state is not None
    assert np.allclose(current_state, psi, atol=1e-7)


def test_gpu_acceleration(gui: QuantumSimulationGUI) -> None:
    """Test GPU-accelerated quantum state calculations."""
    gl_widget = gui.gl_widget

    # Skip test if OpenGL context is not valid
    if not gl_widget.context().isValid():
        pytest.skip("OpenGL context not valid")

    accelerator = gl_widget.gpu_accelerator

    # Create test state
    state = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    state = state / np.sqrt(np.sum(np.abs(state) ** 2))

    # Test universal oneness transformation
    transformed_state = accelerator.apply_universal_oneness(state, 0.5)
    assert transformed_state.shape == state.shape
    assert np.isclose(np.sum(np.abs(transformed_state) ** 2), 1.0, atol=1e-7)

    # Test cleanup
    try:
        accelerator.cleanup()
    except Exception as e:
        logger.warning(f"GPU cleanup warning: {str(e)}")


def test_state_buffer(gui: QuantumSimulationGUI) -> None:
    """Test quantum state buffer functionality."""
    gl_widget = gui.gl_widget

    # Skip test if OpenGL context is not valid
    if not gl_widget.context().isValid():
        pytest.skip("OpenGL context not valid")

    buffer = gl_widget.state_buffer

    # Test state history management
    state1 = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    state2 = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)

    buffer.add_state(state1, timestamp=0.0)
    buffer.add_state(state2, timestamp=1.0)

    # Test interpolation
    interpolated = buffer.get_state_at_time(0.5)
    assert interpolated is not None
    assert interpolated.shape == state1.shape


def test_ui_controls(gui: QuantumSimulationGUI) -> None:
    """Test UI controls and rendering mode switches."""
    gl_widget = gui.gl_widget

    # Skip test if OpenGL context is not valid
    if not gl_widget.context().isValid():
        pytest.skip("OpenGL context not valid")

    # Test rendering mode selection
    QTest.mouseClick(gui.surface_mode, Qt.LeftButton)
    assert gui.surface_mode.isChecked()

    QTest.mouseClick(gui.volumetric_mode, Qt.LeftButton)
    assert gui.volumetric_mode.isChecked()

    QTest.mouseClick(gui.raytracing_mode, Qt.LeftButton)
    assert gui.raytracing_mode.isChecked()


def test_error_handling(gui: QuantumSimulationGUI) -> None:
    """Test error handling and validation."""
    gl_widget = gui.gl_widget

    # Skip test if OpenGL context is not valid
    if not gl_widget.context().isValid():
        pytest.skip("OpenGL context not valid")

    # Test invalid state handling
    with pytest.raises(ValueError):
        gl_widget.set_quantum_state(None)

    # Test invalid dimensions
    with pytest.raises(ValueError):
        gl_widget.set_quantum_state(np.random.rand(50))


@pytest.mark.benchmark
def test_performance(gui: QuantumSimulationGUI) -> None:
    """Test visualization performance under load."""
    gl_widget = gui.gl_widget

    # Skip test if OpenGL context is not valid
    if not gl_widget.context().isValid():
        pytest.skip("OpenGL context not valid")

    # Create large state changes
    for _ in range(10):
        state = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
        state = state / np.sqrt(np.sum(np.abs(state) ** 2))

        # Measure update time
        import time

        start_time = time.time()
        gl_widget.set_quantum_state(state)
        update_time = time.time() - start_time

        # Verify performance
        assert update_time < 0.1, f"State update took {update_time:.3f}s, exceeding 100ms threshold"

        # Allow time for GPU operations to complete
        gl_widget.context().swapBuffers(gl_widget.context().surface())


@pytest.mark.rendering
def test_visualization_modes(gui: QuantumSimulationGUI) -> None:
    """Test different visualization modes and effects."""
    gl_widget = gui.gl_widget

    # Skip test if OpenGL context is not valid
    if not gl_widget.context().isValid():
        pytest.skip("OpenGL context not valid")

    # Test surface plot mode
    gui.surface_mode.setChecked(True)
    gl_widget.update()
    gl_widget.context().swapBuffers(gl_widget.context().surface())

    # Test volumetric mode
    gui.volumetric_mode.setChecked(True)
    gl_widget.update()
    gl_widget.context().swapBuffers(gl_widget.context().surface())

    # Test ray tracing mode
    gui.raytracing_mode.setChecked(True)
    gl_widget.update()
    gl_widget.context().swapBuffers(gl_widget.context().surface())

    # Test glow effect if available
    if hasattr(gui, "glow_effect"):
        gui.glow_effect.setChecked(True)
        gl_widget.update()
        gl_widget.context().swapBuffers(gl_widget.context().surface())

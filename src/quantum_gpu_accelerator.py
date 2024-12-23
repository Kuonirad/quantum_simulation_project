"""
GPU-accelerated quantum state calculations for enhanced visualization performance.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyopencl as cl
import pyopencl.array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPUContext:
    """Manages OpenCL context and command queue."""

    context: cl.Context
    queue: cl.CommandQueue
    program: cl.Program


class QuantumGPUAccelerator:
    """Handles GPU-accelerated quantum state calculations."""

    # OpenCL kernel for quantum state operations
    KERNEL_SOURCE = """
    // Complex number operations
    typedef struct { float x; float y; } cfloat_t;

    cfloat_t cmult(cfloat_t a, cfloat_t b) {
        cfloat_t result;
        result.x = a.x * b.x - a.y * b.y;
        result.y = a.x * b.y + a.y * b.x;
        return result;
    }

    __kernel void apply_universal_oneness(
        __global const cfloat_t* state_in,
        __global cfloat_t* state_out,
        const float oneness_strength,
        const int size
    ) {
        int gid = get_global_id(0);
        if (gid < size) {
            cfloat_t state = state_in[gid];
            // Apply universal oneness transformation
            float magnitude = sqrt(state.x * state.x + state.y * state.y);
            float phase = atan2(state.y, state.x);

            // Modify phase based on oneness strength
            phase = phase * (1.0f - oneness_strength);

            // Update state with modified phase
            state_out[gid].x = magnitude * cos(phase);
            state_out[gid].y = magnitude * sin(phase);
        }
    }

    __kernel void apply_quantum_resonance(
        __global const cfloat_t* state_in,
        __global cfloat_t* state_out,
        const float resonance_freq,
        const float time,
        const int size
    ) {
        int gid = get_global_id(0);
        if (gid < size) {
            cfloat_t state = state_in[gid];
            float magnitude = sqrt(state.x * state.x + state.y * state.y);
            float phase = atan2(state.y, state.x);

            // Apply resonance effect
            float resonance_phase = resonance_freq * time;
            phase += resonance_phase;

            // Update state
            state_out[gid].x = magnitude * cos(phase);
            state_out[gid].y = magnitude * sin(phase);
        }
    }

    __kernel void apply_volumetric_rendering(
        __global const cfloat_t* state_in,
        __global float* density_out,
        __global float* opacity_out,
        const int size,
        const float iso_value,
        const float opacity_scale
    ) {
        int gid = get_global_id(0);
        if (gid < size) {
            cfloat_t state = state_in[gid];
            float magnitude = sqrt(state.x * state.x + state.y * state.y);

            // Calculate density based on probability amplitude
            density_out[gid] = magnitude * magnitude;

            // Calculate opacity using transfer function
            float opacity = (magnitude > iso_value) ?
                opacity_scale * (magnitude - iso_value) : 0.0f;
            opacity_out[gid] = clamp(opacity, 0.0f, 1.0f);
        }
    }

    __kernel void apply_ray_tracing(
        __global const cfloat_t* state_in,
        __global float4* color_out,
        const int size,
        const float3 light_pos,
        const float ambient_strength,
        const float diffuse_strength
    ) {
        int gid = get_global_id(0);
        if (gid < size) {
            cfloat_t state = state_in[gid];
            float magnitude = sqrt(state.x * state.x + state.y * state.y);
            float phase = atan2(state.y, state.x);

            // Calculate surface normal from probability gradient
            float3 normal = normalize((float3)(
                state.x, state.y, magnitude
            ));

            // Calculate lighting
            float3 light_dir = normalize(light_pos);
            float diffuse = max(dot(normal, light_dir), 0.0f);

            // Calculate final color with phase-based hue
            float hue = (phase + M_PI) / (2.0f * M_PI);
            float intensity = ambient_strength + diffuse * diffuse_strength;

            color_out[gid] = (float4)(
                hue,
                1.0f,  // saturation
                intensity,
                magnitude * magnitude  // alpha based on probability
            );
        }
    }
    """

    def __init__(self):
        """Initialize GPU accelerator with OpenCL context."""
        try:
            # Get OpenCL platform and device
            platform = cl.get_platforms()[0]

            # Try GPU first, fall back to CPU if necessary
            try:
                device = platform.get_devices(device_type=cl.device_type.GPU)[0]
                logger.info("Using GPU device for acceleration")
            except cl.RuntimeError as e:
                logger.warning("No GPU found, falling back to CPU device: %s", str(e))
                device = platform.get_devices(device_type=cl.device_type.CPU)[0]

            # Check device capabilities
            if not device.get_info(cl.device_info.IMAGE_SUPPORT):
                logger.warning("Device does not support images, some features may be limited")

            # Create OpenCL context and command queue
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)

            # Build OpenCL program with math precision pragmas
            self.program = cl.Program(
                self.context, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" + self.KERNEL_SOURCE
            ).build()

            logger.info(f"GPU Accelerator initialized on {device.name}")
            self.gpu_context = GPUContext(self.context, self.queue, self.program)

        except Exception as e:
            logger.error("Failed to initialize GPU Accelerator: %s", str(e))
            raise RuntimeError("GPU initialization failed") from e

    def apply_universal_oneness(self, quantum_state: np.ndarray, oneness_strength: float) -> np.ndarray:
        """
        Apply universal oneness transformation using GPU acceleration.

        Args:
            quantum_state: Complex numpy array representing quantum state
            oneness_strength: Strength of the universal oneness effect (0 to 1)

        Returns:
            Modified quantum state array
        """
        try:
            # Prepare input and output buffers
            state_g = cl.array.to_device(self.queue, quantum_state.astype(np.complex64))
            result_g = cl.array.empty_like(state_g)

            # Execute kernel
            self.program.apply_universal_oneness(
                self.queue,
                quantum_state.shape,
                None,
                state_g.data,
                result_g.data,
                np.float32(oneness_strength),
                np.int32(quantum_state.size),
            )

            return result_g.get()

        except cl.RuntimeError as e:
            logger.error("GPU computation error in universal oneness: %s", str(e))
            raise RuntimeError("Universal oneness computation failed") from e

    def apply_quantum_resonance(self, quantum_state: np.ndarray, resonance_freq: float, time: float) -> np.ndarray:
        """
        Apply quantum resonance transformation using GPU acceleration.

        Args:
            quantum_state: Complex numpy array representing quantum state
            resonance_freq: Frequency of quantum resonance
            time: Current simulation time

        Returns:
            Modified quantum state array
        """
        try:
            # Prepare input and output buffers
            state_g = cl.array.to_device(self.queue, quantum_state.astype(np.complex64))
            result_g = cl.array.empty_like(state_g)

            # Execute kernel
            self.program.apply_quantum_resonance(
                self.queue,
                quantum_state.shape,
                None,
                state_g.data,
                result_g.data,
                np.float32(resonance_freq),
                np.float32(time),
                np.int32(quantum_state.size),
            )

            return result_g.get()

        except cl.RuntimeError as e:
            logger.error("GPU computation error in quantum resonance: %s", str(e))
            raise RuntimeError("Quantum resonance computation failed") from e

    def apply_volumetric_rendering(
        self, quantum_state: np.ndarray, iso_value: float = 0.5, opacity_scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply volumetric rendering to quantum state.

        Args:
            quantum_state: Complex numpy array representing quantum state
            iso_value: Threshold for density calculation
            opacity_scale: Scale factor for opacity values

        Returns:
            Tuple of (density array, opacity array)
        """
        try:
            # Prepare input and output buffers
            state_g = cl.array.to_device(self.queue, quantum_state.astype(np.complex64))
            density_g = cl.array.empty(quantum_state.shape, dtype=np.float32)
            opacity_g = cl.array.empty(quantum_state.shape, dtype=np.float32)

            # Execute kernel
            self.program.apply_volumetric_rendering(
                self.queue,
                quantum_state.shape,
                None,
                state_g.data,
                density_g.data,
                opacity_g.data,
                np.int32(quantum_state.size),
                np.float32(iso_value),
                np.float32(opacity_scale),
            )

            return density_g.get(), opacity_g.get()

        except cl.RuntimeError as e:
            logger.error("GPU computation error in volumetric rendering: %s", str(e))
            raise RuntimeError("Volumetric rendering computation failed") from e

    def apply_ray_tracing(
        self,
        quantum_state: np.ndarray,
        light_position: np.ndarray = None,
        ambient_strength: float = 0.3,
        diffuse_strength: float = 0.7,
    ) -> np.ndarray:
        """
        Apply ray tracing to quantum state for advanced lighting effects.

        Args:
            quantum_state: Complex numpy array representing quantum state
            light_position: 3D vector representing light source position
            ambient_strength: Strength of ambient lighting (0 to 1)
            diffuse_strength: Strength of diffuse lighting (0 to 1)

        Returns:
            RGBA color array for each point in the quantum state
        """
        if light_position is None:
            light_position = np.array([1.0, 1.0, 1.0])
        """
        Apply ray tracing to quantum state for advanced lighting effects.

        Args:
            quantum_state: Complex numpy array representing quantum state
            light_position: 3D vector representing light source position
            ambient_strength: Strength of ambient lighting (0 to 1)
            diffuse_strength: Strength of diffuse lighting (0 to 1)

        Returns:
            RGBA color array for each point in the quantum state
        """
        try:
            # Prepare input and output buffers
            state_g = cl.array.to_device(self.queue, quantum_state.astype(np.complex64))
            color_g = cl.array.empty(quantum_state.shape + (4,), dtype=np.float32)

            # Execute kernel
            self.program.apply_ray_tracing(
                self.queue,
                quantum_state.shape,
                None,
                state_g.data,
                color_g.data,
                np.int32(quantum_state.size),
                light_position.astype(np.float32),
                np.float32(ambient_strength),
                np.float32(diffuse_strength),
            )

            return color_g.get()

        except cl.RuntimeError as e:
            logger.error("GPU computation error in ray tracing: %s", str(e))
            raise RuntimeError("Ray tracing computation failed") from e

    def cleanup(self):
        """Release GPU resources."""
        try:
            # Ensure all commands are completed
            self.queue.finish()

            # Release OpenCL resources
            self.program = None
            self.queue = None
            self.context = None
            logger.info("GPU resources cleaned up successfully")
        except Exception as e:
            logger.error("Error during GPU cleanup: %s", str(e))
            raise RuntimeError("GPU cleanup failed") from e

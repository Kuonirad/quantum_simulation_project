"""
Circular buffer implementation for quantum state history management.
"""
from collections import deque
import numpy as np
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumStateBuffer:
    """Manages quantum state history with circular buffer implementation."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize quantum state buffer.

        Args:
            max_size: Maximum number of states to store in history
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        logger.info(f"Initialized quantum state buffer with max size {max_size}")

    def add_state(self, state: np.ndarray, timestamp: float):
        """
        Add quantum state to buffer.

        Args:
            state: Quantum state array
            timestamp: Time of state measurement
        """
        self.buffer.append(state.copy())
        self.timestamps.append(timestamp)

    def get_state_at_time(self, target_time: float) -> Optional[np.ndarray]:
        """
        Get interpolated state at specified time.

        Args:
            target_time: Target timestamp

        Returns:
            Interpolated quantum state or None if time out of range
        """
        if not self.timestamps:
            return None

        # Find nearest timestamps
        times = np.array(self.timestamps)
        idx = np.searchsorted(times, target_time)

        if idx == 0:
            return self.buffer[0]
        if idx == len(times):
            return self.buffer[-1]

        # Interpolate between states
        t0, t1 = times[idx-1], times[idx]
        s0, s1 = self.buffer[idx-1], self.buffer[idx]

        # Linear interpolation weight
        w = (target_time - t0) / (t1 - t0)

        return s0 * (1 - w) + s1 * w

    def get_state_range(
        self,
        start_time: float,
        end_time: float
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Get states within specified time range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Tuple of (states, timestamps) within range
        """
        if not self.timestamps:
            return [], []

        times = np.array(self.timestamps)
        mask = (times >= start_time) & (times <= end_time)

        selected_states = [self.buffer[i] for i in np.where(mask)[0]]
        selected_times = times[mask].tolist()

        return selected_states, selected_times

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.timestamps.clear()
        logger.info("Cleared quantum state buffer")

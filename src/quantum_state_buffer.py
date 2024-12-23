"""
Circular buffer implementation for quantum state history management with thread safety.
"""

import logging
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QMutex, QMutexLocker, QObject, pyqtSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumStateBuffer(QObject):
    """Manages quantum state history with thread-safe circular buffer implementation."""

    # Qt signals for state updates
    state_updated = pyqtSignal(object)  # Emits new state
    buffer_cleared = pyqtSignal()  # Emits when buffer is cleared

    def __init__(self, max_size: int = 1000):
        """
        Initialize quantum state buffer.

        Args:
            max_size: Maximum number of states to store in history
        """
        super().__init__()
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self._mutex = QMutex()
        logger.info(f"Initialized quantum state buffer with max size {max_size}")

    def add_state(self, state: np.ndarray, timestamp: float):
        """
        Add quantum state to buffer in a thread-safe manner.

        Args:
            state: Quantum state array
            timestamp: Time of state measurement
        """
        with QMutexLocker(self._mutex):
            self.buffer.append(state.copy())
            self.timestamps.append(timestamp)
            # Emit signal with copy of state to prevent modification during signal handling
            self.state_updated.emit(state.copy())

    def get_state_at_time(self, target_time: float) -> Optional[np.ndarray]:
        """
        Get interpolated state at specified time in a thread-safe manner.

        Args:
            target_time: Target timestamp

        Returns:
            Interpolated quantum state or None if time out of range
        """
        with QMutexLocker(self._mutex):
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
            t0, t1 = times[idx - 1], times[idx]
            s0, s1 = self.buffer[idx - 1], self.buffer[idx]

            # Linear interpolation weight
            w = (target_time - t0) / (t1 - t0)

            return s0 * (1 - w) + s1 * w

    def get_state_range(self, start_time: float, end_time: float) -> Tuple[List[np.ndarray], List[float]]:
        """
        Get states within specified time range in a thread-safe manner.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Tuple of (states, timestamps) within range
        """
        with QMutexLocker(self._mutex):
            if not self.timestamps:
                return [], []

            times = np.array(self.timestamps)
            mask = (times >= start_time) & (times <= end_time)

            selected_states = [self.buffer[i] for i in np.where(mask)[0]]
            selected_times = times[mask].tolist()

            return selected_states, selected_times

    def clear(self):
        """Clear the buffer in a thread-safe manner."""
        with QMutexLocker(self._mutex):
            self.buffer.clear()
            self.timestamps.clear()
            self.buffer_cleared.emit()
            logger.info("Cleared quantum state buffer")

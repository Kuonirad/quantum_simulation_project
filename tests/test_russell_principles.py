import unittest
import numpy as np
import torch
from scipy.linalg import expm
from src.advanced_quantum_simulation import walter_russell_principles, QHRModel

class TestRussellPrinciples(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.H0 = np.array([[1, 0], [0, -1]])  # Simple two-level system
        self.t = 0.0
        self.chi = 0.1
        self.omega = 1.0
        self.alpha = 0.5

    def test_cosmic_duality_operator(self):
        """Test if cosmic duality operator is unitary"""
        C = expm(1j * self.chi * self.H0)
        # Check if C†C = I (unitary property)
        identity = np.eye(2)
        product = C.conj().T @ C
        np.testing.assert_array_almost_equal(product, identity)

    def test_rbi_operator(self):
        """Test RBI operator properties"""
        times = np.linspace(0, 2*np.pi, 100)
        values = [self.alpha * np.sin(self.omega * t) for t in times]
        # Check periodicity
        np.testing.assert_array_almost_equal(values[0], values[-1], decimal=5)

    def test_qhr_model(self):
        """Test QHR model forward pass"""
        input_size = 10
        hidden_size = 20
        output_size = 5
        batch_size = 32
        seq_length = 100

        model = QHRModel(input_size, hidden_size, output_size)
        x = torch.randn(batch_size, seq_length, input_size)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, output_size))

    def test_enhanced_hamiltonian(self):
        """Test enhanced Hamiltonian properties"""
        H_enhanced = self.H0 + self.alpha * np.sin(self.omega * self.t) * np.eye(2)

        # Check Hermiticity
        np.testing.assert_array_almost_equal(
            H_enhanced, H_enhanced.conj().T
        )

if __name__ == '__main__':
    unittest.main()

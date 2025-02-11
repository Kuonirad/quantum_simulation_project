import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers import Backend
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit_aer import AerSimulator
from qiskit.circuit import ClassicalRegister
from scipy.optimize import curve_fit

class ErrorMitigator:
    """Implements comprehensive error mitigation for quantum drug discovery."""
    
    def __init__(self, backend: Backend):
        self.backend = backend
        self.meas_fitter = None
        # Extended noise scales to capture non-linear behavior
        self.zne_scales = [1.0, 2.0, 3.0, 4.0]
        
    def calculate_aic(self, n_params: int, log_likelihood: float) -> float:
        """Compute Akaike Information Criterion.
        
        Args:
            n_params: Number of parameters in the model
            log_likelihood: Log-likelihood of the model fit
            
        Returns:
            float: AIC value (lower is better)
        """
        return 2 * n_params - 2 * log_likelihood
        
    def calculate_bic(self, n_params: int, n_samples: int, log_likelihood: float) -> float:
        """Compute Bayesian Information Criterion.
        
        Args:
            n_params: Number of parameters in the model
            n_samples: Number of data points
            log_likelihood: Log-likelihood of the model fit
            
        Returns:
            float: BIC value (lower is better)
        """
        return n_params * np.log(n_samples) - 2 * log_likelihood
        
    def compute_log_likelihood(self, residuals: np.ndarray, sigma: float) -> float:
        """Compute log-likelihood assuming Gaussian noise.
        
        Args:
            residuals: Array of residuals (observed - predicted)
            sigma: Standard deviation of noise
            
        Returns:
            float: Log-likelihood value
        """
        n = len(residuals)
        return -n/2 * np.log(2*np.pi*sigma**2) - np.sum(residuals**2)/(2*sigma**2)
        
    def calibrate_measurement(self, num_qubits: int) -> None:
        """Performs measurement calibration using modern Qiskit primitives."""
        # Initialize backend
        self.backend = AerSimulator(method='statevector')
        
        # Create calibration circuits for all basis states
        self.cal_circuits = []
        self.state_labels = []
        
        for i in range(2**num_qubits):
            # Create circuit for basis state |i⟩
            qc = QuantumCircuit(num_qubits, num_qubits)  # Include classical bits
            bin_str = format(i, f'0{num_qubits}b')
            for j, bit in enumerate(bin_str):
                if bit == '1':
                    qc.x(j)
            qc.measure_all()
            
            self.cal_circuits.append(qc)
            self.state_labels.append(bin_str)
        
        # Run calibration circuits with higher shot count for better statistics
        transpiled_calibs = transpile(self.cal_circuits, backend=self.backend)
        self.cal_results = self.backend.run(transpiled_calibs, shots=16384).result()
        
        # Pre-compute calibration matrix for efficiency
        num_states = len(self.state_labels)
        self.cal_matrix = np.zeros((num_states, num_states))
        
        for i, circuit in enumerate(self.cal_circuits):
            result_counts = self.cal_results.get_counts(circuit)
            total_shots = sum(result_counts.values())
            
            for j, label in enumerate(self.state_labels):
                self.cal_matrix[i, j] = result_counts.get(label, 0) / total_shots
                
        # Pre-compute inverse calibration matrix
        try:
            self.cal_matrix_inv = np.linalg.inv(self.cal_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            self.cal_matrix_inv = np.linalg.pinv(self.cal_matrix)
        
    def apply_zero_noise_extrapolation(
        self,
        circuit: QuantumCircuit,
        observable: Union[np.ndarray, SparsePauliOp],
        backend: Optional[Backend] = None,
        shots: int = 8192
    ) -> Tuple[float, float]:
        """Implements Richardson extrapolation for zero-noise limit.
        
        Args:
            circuit: Circuit to execute
            observable: Observable to measure (SparsePauliOp or numpy array)
            backend: Optional backend to use (defaults to self.backend)
            shots: Number of shots for execution
            
        Returns:
            Tuple of (extrapolated value, error estimate)
        """
        results = []
        
        # Create a copy of the circuit to avoid modifying the original
        circuit_copy = circuit.copy()
        
        # Add classical bits and measurement if not present
        if circuit_copy.num_clbits == 0:
            creg = ClassicalRegister(circuit_copy.num_qubits, name='meas')
            circuit_copy.add_register(creg)
            circuit_copy.measure_all()
        
        # Use provided backend or default
        if backend is None:
            backend = self.backend
        
        # Execute circuit at different noise scales with increased shot count for 4x
        for scale in self.zne_scales:
            # Create scaled circuit
            scaled_circuit = self._scale_noise(circuit_copy, scale)
            
            # Add save_statevector instruction
            scaled_circuit.save_state()
            
            # Increase shots for higher noise scales to maintain precision
            scale_shots = shots * (2 if scale >= 4.0 else 1)
            
            # Execute using AerSimulator
            transpiled_circuit = transpile(scaled_circuit, backend)
            job = backend.run(transpiled_circuit)
            result = job.result()
            
            # Get counts from result
            counts = result.get_counts()
            
            # Convert observable to array if needed
            if isinstance(observable, SparsePauliOp):
                obs_array = observable.to_matrix().diagonal().real
            else:
                obs_array = observable
            
            expectation = self._compute_expectation(counts, obs_array)
            results.append(expectation)
            
        # Perform multi-model extrapolation
        zero_noise_value, best_model, fit_data = self._extrapolate_to_zero(
            self.zne_scales,
            results
        )
        
        # Estimate error using model-specific method
        error_estimate = self._estimate_zne_error(results, best_model, fit_data)
        
        # Log extrapolation details for analysis
        print(f"Selected model: {best_model}")
        print(f"Fit residuals: {np.std(fit_data[best_model]['residuals']):.2e}")
        print(f"Total error estimate: {error_estimate:.2e}")
        
        return zero_noise_value, error_estimate
    
    def correct_measurement_errors(
        self,
        counts: Dict[str, int]
    ) -> Dict[str, float]:
        """Applies measurement error correction using calibration data."""
        if not hasattr(self, 'cal_results'):
            raise ValueError("Measurement calibration not performed")
            
        # Use pre-computed calibration matrices
        num_states = len(self.state_labels)
        
        # Convert input counts to probabilities
        total_shots = max(sum(counts.values()), 1)  # Avoid division by zero
        raw_probs = np.zeros(num_states)
        for i, label in enumerate(self.state_labels):
            raw_probs[i] = counts.get(label, 0) / total_shots
        
        # Apply correction using pre-computed inverse matrix
        mitigated_probs = self.cal_matrix_inv @ raw_probs
        
        # Convert back to counts format
        mitigated_counts = {}
        for label, prob in zip(self.state_labels, mitigated_probs):
            # Ensure probabilities are physical and non-zero
            prob = max(1e-10, min(1, prob.real))
            mitigated_counts[label] = max(1, int(prob * total_shots))
            
        return mitigated_counts
    
    def _scale_noise(
        self,
        circuit: QuantumCircuit,
        scale: float
    ) -> QuantumCircuit:
        """Scales circuit noise by repeating gates."""
        if scale == 1.0:
            return circuit.copy()
            
        # Create new circuit with same registers
        scaled = QuantumCircuit(
            *circuit.qregs,
            *circuit.cregs,
            name=f"{circuit.name}_scaled_{scale}"
        )
        
        # Repeat each gate operation to scale noise
        for instruction in circuit.data:
            # Number of repetitions for this scale
            reps = int(scale)
            
            # Add repeated gates
            for _ in range(reps):
                # Use named attributes instead of indexing
                scaled.append(
                    instruction.operation,
                    instruction.qubits,
                    instruction.clbits
                )
                
        return scaled
    
    def _extrapolate_to_zero(
        self,
        scales: List[float],
        values: List[float]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Performs multi-model extrapolation with AIC/BIC selection.
        
        Args:
            scales: List of noise scaling factors
            values: Corresponding measurement values
            
        Returns:
            Tuple of (best_estimate, model_name, fit_data)
        """
        models = {}
        n_samples = len(scales)
        
        # Ensure non-zero values for numerical stability
        values = np.array(values)
        values = np.where(np.abs(values) < 1e-10, 1e-10, values)
        
        # Estimate noise level for likelihood calculation
        sigma = max(np.std(values) / np.sqrt(n_samples), 1e-10)
        
        # Try different models
        for model_name, n_params in [
            ('linear', 2),
            ('quadratic', 3),
            ('cubic', 4),
            ('exponential', 3)
        ]:
            try:
                if model_name == 'exponential':
                    # Exponential model: a * exp(-k * x) + b
                    popt, pcov = curve_fit(
                        lambda x, a, k, b: a * np.exp(-k * x) + b,
                        scales, values, sigma=sigma,
                        p0=[1.0, 0.1, values[-1]],  # Better initial guess
                        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])
                    )
                    residuals = values - (popt[0] * np.exp(-popt[1] * np.array(scales)) + popt[2])
                    zero_noise_value = popt[2]  # b is the zero-noise limit
                else:
                    # Polynomial models with weighted fit
                    weights = 1.0 / (sigma * np.ones_like(scales))
                    popt = np.polyfit(scales, values, n_params-1, w=weights)
                    residuals = values - np.polyval(popt, scales)
                    zero_noise_value = np.polyval(popt, 0.0)
                
                # Compute information criteria with robust likelihood
                log_likelihood = self.compute_log_likelihood(residuals, sigma)
                aic = self.calculate_aic(n_params, log_likelihood)
                bic = self.calculate_bic(n_params, n_samples, log_likelihood)
                
                models[model_name] = {
                    'params': popt,
                    'zero_value': zero_noise_value,
                    'aic': aic,
                    'bic': bic,
                    'residuals': residuals,
                    'log_likelihood': log_likelihood
                }
                
            except (np.linalg.LinAlgError, RuntimeError):
                continue
        
        if not models:
            raise ValueError("All extrapolation models failed")
        
        # Select best model using AIC and additional criteria
        aic_scores = {name: data['aic'] for name, data in models.items()}
        min_aic = min(aic_scores.values())
        
        # Consider models within AIC difference threshold
        threshold = 2.0  # Models within 2 AIC units are considered comparable
        comparable_models = [
            name for name, data in models.items()
            if data['aic'] - min_aic < threshold
        ]
        
        # Prefer quadratic model if it has good fit
        if 'quadratic' in models:
            quadratic_residuals = models['quadratic']['residuals']
            quadratic_rmse = np.sqrt(np.mean(quadratic_residuals**2))
            quadratic_r2 = 1 - np.sum(quadratic_residuals**2) / np.sum((values - np.mean(values))**2)
            
            # If quadratic model has good fit and is comparable, prefer it
            if quadratic_rmse < 0.1 and quadratic_r2 > 0.95:
                model_name = 'quadratic'
            else:
                # Choose model with minimum AIC among comparable models
                model_name = min(comparable_models, key=lambda x: models[x]['aic'])
        else:
            # Choose model with minimum AIC among comparable models
            model_name = min(comparable_models, key=lambda x: models[x]['aic'])
            
        model_data = models[model_name]
        
        # Log model selection details
        print(f"Selected model: {model_name}")
        print(f"AIC values: {[(name, data['aic']) for name, data in models.items()]}")
        print(f"Residual std: {np.std(model_data['residuals']):.2e}")
        
        return model_data['zero_value'], model_name, models
    
    def _estimate_zne_error(
        self,
        values: List[float],
        model: str,
        fit_data: Dict[str, Any]
    ) -> float:
        """Estimates error in zero-noise extrapolation using model-specific methods.
        
        Args:
            values: List of measured values
            model: Name of extrapolation model used
            fit_data: Dictionary containing fit parameters and residuals
            
        Returns:
            Estimated error in extrapolated value
        """
        # Base statistical error from measurements
        statistical_error = np.std(values) / np.sqrt(len(values))
        
        # Model-specific systematic error
        residuals = fit_data[model]['residuals']
        systematic_error = np.std(residuals)
        
        # Combine errors (assuming independence)
        total_error = np.sqrt(statistical_error**2 + systematic_error**2)
        
        return total_error
    
    def execute_with_mitigation(
        self,
        func: Callable,
        circuit: QuantumCircuit,
        observable: Optional[SparsePauliOp] = None
    ) -> float:
        """Execute a quantum operation with error mitigation.
        
        Args:
            func: Callable that executes the quantum operation
            circuit: Quantum circuit to execute
            observable: Observable to measure (optional)
            
        Returns:
            float: Mitigated result
        """
        # First execute without mitigation
        raw_result = func()
        
        # Apply zero-noise extrapolation if circuit and observable provided
        if circuit is not None and observable is not None:
            # Convert SparsePauliOp to array form for ZNE
            if isinstance(observable, SparsePauliOp):
                obs_array = observable.to_matrix().diagonal().real
            else:
                obs_array = observable
                
            mitigated_result, _ = self.apply_zero_noise_extrapolation(
                circuit,
                observable,  # Pass SparsePauliOp directly
                backend=self.backend
            )
            return mitigated_result
        
        return raw_result


    def _compute_expectation(
        self,
        counts: Dict[str, int],
        observable: Union[np.ndarray, SparsePauliOp]
    ) -> float:
        """Computes expectation value from measurement counts.
        
        Args:
            counts: Dictionary of measurement counts
            observable: Observable operator (SparsePauliOp) or diagonal array
            
        Returns:
            float: Expectation value
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        # Convert SparsePauliOp to diagonal array if needed
        if isinstance(observable, SparsePauliOp):
            observable = observable.to_matrix().diagonal().real
            
        expectation = 0.0
        n_qubits = len(next(iter(counts)).replace(" ", ""))
        obs_size = 2**n_qubits
        
        # Pad or truncate observable to match number of qubits
        if len(observable) < obs_size:
            observable = np.pad(observable, (0, obs_size - len(observable)))
        elif len(observable) > obs_size:
            observable = observable[:obs_size]
        
        for bitstring, count in counts.items():
            # Convert bitstring to state index
            state_idx = int(bitstring.replace(" ", ""), 2)
            # Add contribution weighted by counts
            expectation += (observable[state_idx] * count) / total_shots
            
        return expectation

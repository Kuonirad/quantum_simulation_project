import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, Estimator
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
from qiskit.algorithms.optimizers import COBYLA, SPSA
from scipy.linalg import eigh

from .error_mitigation import ErrorMitigator
from .molecular_hamiltonian import MolecularHamiltonian

class HybridSolver:
    """Implements hybrid quantum-classical optimization loop for molecular simulations."""
    
    def __init__(self, backend, error_mitigator: ErrorMitigator):
        self.backend = backend
        self.error_mitigator = error_mitigator
        self.hamiltonian_constructor = MolecularHamiltonian()
        
    def optimize_molecule(
        self,
        molecular_data: Dict,
        method: str = 'vqe',
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        ansatz: Optional[QuantumCircuit] = None,
        estimator: Optional[StatevectorEstimator] = None,
        hamiltonian: Optional[SparsePauliOp] = None
    ) -> Dict:
        """
        Main optimization loop for molecular ground state and excited states.
        
        Args:
            molecular_data: Dictionary containing molecular integral data
            method: 'vqe', 'qse', or 'qaoa'
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            ansatz: Optional quantum circuit for the ansatz
            estimator: Optional StatevectorEstimator instance
            hamiltonian: Optional SparsePauliOp Hamiltonian
        """
        # Use provided Hamiltonian or construct from molecular data
        if hamiltonian is None:
            hamiltonian = self.hamiltonian_constructor.construct_electronic_hamiltonian(
                molecular_data['one_body_integrals'],
                molecular_data['two_body_integrals'],
                molecular_data['n_electrons'],
                molecular_data['n_orbitals']
            )
        
        if method == 'vqe':
            result = self._run_vqe_loop(
                hamiltonian,
                max_iterations,
                tolerance,
                ansatz=ansatz,
                estimator=estimator
            )
        elif method == 'qse':
            result = self._run_qse_loop(
                hamiltonian,
                max_iterations,
                tolerance
            )
        elif method == 'qaoa':
            result = self._run_qaoa_loop(
                hamiltonian,
                max_iterations,
                tolerance
            )
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return result
    
    def _run_vqe_loop(
        self,
        hamiltonian: SparsePauliOp,
        max_iterations: int,
        tolerance: float,
        ansatz: Optional[QuantumCircuit] = None,
        estimator: Optional[StatevectorEstimator] = None
    ) -> Dict:
        """Implements VQE optimization loop with error mitigation."""
        # Initialize optimizer with conservative settings
        self._convergence_history = []  # Reset history
        
        # Initialize convergence history
        self._convergence_history = []
        
        # Initialize ansatz parameters
        initial_params = self._initialize_ansatz_parameters(ansatz)
        
        # Define objective function with error mitigation
        def objective(params):
            # Prepare circuit with current parameters
            circuit = ansatz.copy()
            # Assign parameters to the ansatz circuit
            parameter_dict = dict(zip(circuit.parameters, params))
            circuit = circuit.assign_parameters(parameter_dict)
            
            # Execute with error mitigation
            job = estimator.run([(circuit, [hamiltonian])])
            result = job.result()
            energy = result[0].data.evs[0].real
            
            # Apply error mitigation
            mitigated_energy = self.error_mitigator.execute_with_mitigation(
                lambda: energy,
                circuit,
                hamiltonian
            )
            
            # Record point in convergence history
            self._convergence_history.append({
                'iteration': len(self._convergence_history),
                'energy': mitigated_energy.real,
                'std': 0.0  # No step size information
            })
            
            return mitigated_energy.real
        
        # Record initial point
        initial_energy = objective(initial_params)
        
        def callback(x):
            """Callback to track optimization progress."""
            # Stop if we have enough iterations
            if len(self._convergence_history) >= max_iterations:
                return True  # Stop optimization
            return False  # Continue optimization
            
        optimizer = COBYLA(
            maxiter=max_iterations,
            tol=tolerance,
            rhobeg=0.1,  # Initial trust region radius
            callback=callback  # Add callback to optimizer
        )
        
        # Use provided ansatz or create default
        if ansatz is None:
            n_qubits = self.hamiltonian_constructor.n_orbitals * 2
            ansatz = QuantumCircuit(n_qubits)
            for q in range(n_qubits):
                ansatz.h(q)
        
        # Initialize ansatz parameters with provided ansatz
        initial_params = self._initialize_ansatz_parameters(ansatz)
        
        # Use provided estimator or create default
        if estimator is None:
            estimator = StatevectorEstimator()
        
        # Define objective function with error mitigation
        def objective(params):
            # Prepare circuit with current parameters
            circuit = ansatz.copy()
            # Assign parameters to the ansatz circuit
            parameter_dict = dict(zip(circuit.parameters, params))
            circuit = circuit.assign_parameters(parameter_dict)
            
            # Execute with error mitigation
            job = estimator.run([(circuit, [hamiltonian])])
            result = job.result()
            energy = result[0].data.evs[0].real
            
            # Apply error mitigation
            mitigated_energy = self.error_mitigator.execute_with_mitigation(
                lambda: energy,
                circuit,
                hamiltonian
            )
            
            return mitigated_energy.real
        
        # Run optimization loop with timeout
        import time
        start_time = time.time()
        max_time = 60  # Maximum 60 seconds for optimization
        
        def timeout_checker():
            return time.time() - start_time > max_time
        
        # Run optimization with timeout and debug prints
        try:
            print("Starting VQE optimization...")
            print(f"Initial parameters: {initial_params}")
            print(f"Ansatz circuit:\n{ansatz}")
            print(f"Hamiltonian:\n{hamiltonian}")
            
            # Evaluate initial objective
            initial_energy = objective(initial_params)
            print(f"Initial energy: {initial_energy}")
            
            opt_result = optimizer.minimize(
                fun=objective,
                x0=initial_params
            )
            print("VQE optimization completed successfully")
            
            # Check if timeout occurred
            if timeout_checker():
                print("VQE optimization timed out after 60 seconds")
        except Exception as e:
            print(f"VQE optimization failed: {str(e)}")
            # Return best result so far
            opt_result = type('OptResult', (), {
                'x': initial_params,
                'fun': objective(initial_params)
            })()
        
        # Compute final energy with error mitigation
        final_energy = objective(opt_result.x)
        
        return {
            'energy': final_energy,
            'optimal_parameters': opt_result.x,
            'convergence_history': self._convergence_history
        }
    
    def _run_qse_loop(
        self,
        hamiltonian: SparsePauliOp,
        max_iterations: int,
        tolerance: float
    ) -> Dict:
        """Implements QSE for excited states with error mitigation."""
        # First run VQE to get ground state
        vqe_result = self._run_vqe_loop(
            hamiltonian,
            max_iterations,
            tolerance
        )
        
        # Prepare ground state circuit
        ground_state_circuit = self._prepare_ansatz_circuit(
            vqe_result['optimal_parameters']
        )
        
        # Generate QSE operators
        qse_operators = self._generate_qse_operators(hamiltonian)
        
        # Compute QSE matrix elements with error mitigation
        H_matrix = np.zeros((len(qse_operators), len(qse_operators)), dtype=complex)
        S_matrix = np.zeros_like(H_matrix)
        
        for i, op_i in enumerate(qse_operators):
            for j, op_j in enumerate(qse_operators):
                # Compute matrix elements with error mitigation
                # Use compose for operator multiplication
                H_op = hamiltonian.compose(op_i.adjoint()).compose(op_j)
                H_matrix[i,j] = self.error_mitigator.execute_with_mitigation(
                    self._measure_operator_expectation,
                    ground_state_circuit,
                    H_op
                )
                
                # Use compose for operator multiplication
                S_op = op_i.adjoint().compose(op_j)
                S_matrix[i,j] = self.error_mitigator.execute_with_mitigation(
                    self._measure_operator_expectation,
                    ground_state_circuit,
                    S_op
                )
        
        # Solve generalized eigenvalue problem
        energies, vectors = self._solve_generalized_eigenvalue_problem(
            H_matrix,
            S_matrix
        )
        
        return {
            'ground_state_energy': vqe_result['energy'],
            'excited_state_energies': energies,
            'qse_vectors': vectors
        }
    
    def _run_qaoa_loop(
        self,
        hamiltonian: SparsePauliOp,
        max_iterations: int,
        tolerance: float
    ) -> Dict:
        """Implements QAOA optimization loop with error mitigation."""
        # Initialize QAOA parameters
        initial_params = self._initialize_qaoa_parameters()
        
        # Initialize optimizer
        optimizer = COBYLA(
            maxiter=max_iterations,
            tol=tolerance
        )
        
        # Initialize estimator
        estimator = Estimator()
        
        # Define QAOA objective function with error mitigation
        def qaoa_objective(params):
            # Prepare QAOA circuit
            circuit = self._prepare_qaoa_circuit(
                hamiltonian,
                params
            )
            
            # Execute with estimator and error mitigation
            job = estimator.run([(circuit, [hamiltonian])])
            result = job.result()
            energy = result[0].data.evs[0].real
            
            # Apply error mitigation
            mitigated_energy = self.error_mitigator.execute_with_mitigation(
                lambda: energy,
                circuit,
                hamiltonian
            )
            
            return mitigated_energy.real
        
        # Run optimization
        opt_result = optimizer.minimize(
            fun=qaoa_objective,
            x0=initial_params,
            bounds=None
        )
        
        # Compute final configuration
        final_circuit = self._prepare_qaoa_circuit(
            hamiltonian,
            opt_result.x
        )
        
        final_state = self.error_mitigator.execute_with_mitigation(
            self._get_statevector,
            final_circuit
        )
        
        return {
            'optimal_energy': opt_result.fun,
            'optimal_parameters': opt_result.x,
            'optimal_state': final_state
        }
    
    def _initialize_ansatz_parameters(self, ansatz: Optional[QuantumCircuit] = None) -> np.ndarray:
        """Initializes VQE ansatz parameters."""
        if ansatz is not None:
            # Use number of parameters from provided ansatz
            n_params = len(ansatz.parameters)
            return np.random.randn(n_params)
            
        # Default hardware efficient ansatz parameters
        n_qubits = self.hamiltonian_constructor.n_orbitals * 2
        n_layers = 2
        params_per_layer = n_qubits * 3  # Rx, Ry, Rz rotations
        
        return np.random.randn(n_layers * params_per_layer)
    
    def _initialize_qaoa_parameters(self) -> np.ndarray:
        """Initializes QAOA parameters."""
        p = 2  # Number of QAOA layers
        return np.random.randn(2 * p)  # gamma and beta parameters
    
    def _prepare_ansatz_circuit(
        self,
        parameters: np.ndarray
    ) -> QuantumCircuit:
        """Prepares hardware-efficient ansatz circuit."""
        n_qubits = self.hamiltonian_constructor.n_orbitals * 2
        circuit = QuantumCircuit(n_qubits)
        
        # Initial state preparation
        for q in range(n_qubits):
            circuit.h(q)
        
        # Add parameterized layers
        param_idx = 0
        n_layers = len(parameters) // (n_qubits * 3)
        
        for _ in range(n_layers):
            # Single-qubit rotations
            for q in range(n_qubits):
                circuit.rx(parameters[param_idx], q)
                param_idx += 1
                circuit.ry(parameters[param_idx], q)
                param_idx += 1
                circuit.rz(parameters[param_idx], q)
                param_idx += 1
            
            # Entangling layer
            for q in range(n_qubits - 1):
                circuit.cx(q, q + 1)
            circuit.cx(n_qubits - 1, 0)  # Close the chain
            
        return circuit
    
    def _prepare_qaoa_circuit(
        self,
        hamiltonian: SparsePauliOp,
        parameters: np.ndarray
    ) -> QuantumCircuit:
        """Prepares QAOA circuit."""
        n_qubits = self.hamiltonian_constructor.n_orbitals * 2
        p = len(parameters) // 2
        circuit = QuantumCircuit(n_qubits)
        
        # Initial state preparation
        for q in range(n_qubits):
            circuit.h(q)
        
        # QAOA layers
        for i in range(p):
            gamma, beta = parameters[2*i:2*i+2]
            
            # Problem unitary
            circuit.compose(
                self._problem_unitary(hamiltonian, gamma),
                inplace=True
            )
            
            # Mixing unitary
            for q in range(n_qubits):
                circuit.rx(2 * beta, q)
                
        return circuit
    
    def _problem_unitary(
        self,
        hamiltonian: SparsePauliOp,
        gamma: float
    ) -> QuantumCircuit:
        """Constructs problem unitary for QAOA."""
        n_qubits = self.hamiltonian_constructor.n_orbitals * 2
        circuit = QuantumCircuit(n_qubits)
        
        # Decompose Hamiltonian into Pauli terms
        for pauli_str, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
            # Add corresponding gates for basis change
            for q, p in enumerate(pauli_str):
                if p == 'X':
                    circuit.h(q)
                elif p == 'Y':
                    circuit.sdg(q)
                    circuit.h(q)
                    
            # Add controlled-Z between qubits with non-identity Paulis
            for q1 in range(n_qubits):
                for q2 in range(q1 + 1, n_qubits):
                    if pauli_str[q1] != 'I' and pauli_str[q2] != 'I':
                        circuit.cz(q1, q2)
                        
            # Add phase rotation with coefficient
            circuit.rz(2 * gamma * coeff.real, 0)
            
            # Inverse basis change
            for q, p in enumerate(pauli_str):
                if p == 'X':
                    circuit.h(q)
                elif p == 'Y':
                    circuit.h(q)
                    circuit.s(q)
                    
        return circuit
    
    def _generate_qse_operators(
        self,
        hamiltonian: SparsePauliOp
    ) -> List[SparsePauliOp]:
        """Generates set of operators for QSE."""
        operators = []
        num_qubits = len(hamiltonian.paulis[0])
        
        # Add identity operator
        operators.append(SparsePauliOp(["I" * num_qubits], coefficients=[1.0]))
        
        # Add single-qubit excitation operators
        for i in range(num_qubits):
            # X operator at position i
            x_op = "I" * i + "X" + "I" * (num_qubits - i - 1)
            operators.append(SparsePauliOp([x_op], coefficients=[1.0]))
            
            # Y operator at position i
            y_op = "I" * i + "Y" + "I" * (num_qubits - i - 1)
            operators.append(SparsePauliOp([y_op], coefficients=[1.0]))
            
        return operators
    
    def _solve_generalized_eigenvalue_problem(
        self,
        H_matrix: np.ndarray,
        S_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solves generalized eigenvalue problem for QSE."""
        # Ensure matrices are Hermitian
        H_matrix = (H_matrix + H_matrix.conj().T) / 2
        S_matrix = (S_matrix + S_matrix.conj().T) / 2
        
        # Solve generalized eigenvalue problem
        energies, vectors = np.linalg.eigh(H_matrix, S_matrix)
        
        # Sort by energy
        idx = np.argsort(energies.real)
        energies = energies[idx]
        vectors = vectors[:, idx]
        
        return energies, vectors
    
    def _convergence_callback(
        self,
        n_iter: int,
        params: np.ndarray,
        energy: float,
        std: float,
        nfev: Optional[int] = None
    ) -> None:
        """Callback function to track optimization convergence."""
        if not hasattr(self, '_convergence_history'):
            self._convergence_history = []
            
        self._convergence_history.append({
            'iteration': n_iter,
            'energy': energy,
            'std': std
        })
        
    def _get_statevector(
        self,
        circuit: QuantumCircuit
    ) -> Statevector:
        """Gets the statevector from a quantum circuit."""
        # Create a copy of the circuit without measurements
        sv_circuit = circuit.copy()
        sv_circuit.remove_final_measurements()
        
        # Add instruction to save statevector
        sv_circuit.save_statevector()
        
        # Run on backend
        job = self.backend.run(sv_circuit)
        result = job.result()
        
        # Return statevector
        return Statevector(result.data()['statevector'])

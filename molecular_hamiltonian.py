import numpy as np
from typing import List, Dict, Tuple
from qiskit.quantum_info import SparsePauliOp, Pauli

class MolecularHamiltonian:
    """Constructs and manages molecular Hamiltonians for quantum drug discovery."""
    
    def __init__(self):
        self.n_orbitals = None
        self.n_electrons = None
        
    def construct_electronic_hamiltonian(
        self,
        one_body_integrals: np.ndarray,
        two_body_integrals: np.ndarray,
        n_electrons: int,
        n_orbitals: int
    ) -> SparsePauliOp:
        """
        Constructs electronic structure Hamiltonian from molecular integrals.
        
        Args:
            one_body_integrals: One-electron integrals (kinetic + nuclear attraction)
            two_body_integrals: Two-electron integrals (electron-electron repulsion)
            n_electrons: Number of electrons
            n_orbitals: Number of spatial orbitals
            
        Returns:
            SparsePauliOp: The qubit Hamiltonian in sparse Pauli operator form
        """
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        
        # Convert to spin-orbital basis
        one_body_spin = self._to_spin_orbital_basis(one_body_integrals)
        two_body_spin = self._to_spin_orbital_basis(two_body_integrals)
        
        # Generate fermionic operators
        fermionic_hamiltonian = self._generate_fermionic_hamiltonian(
            one_body_spin,
            two_body_spin
        )
        
        # Transform to qubit operators using Jordan-Wigner
        qubit_hamiltonian = self._jordan_wigner_transform(fermionic_hamiltonian)
        
        # Apply symmetry reduction
        reduced_hamiltonian = self._apply_symmetry_reduction(qubit_hamiltonian)
        
        return reduced_hamiltonian
    
    def _to_spin_orbital_basis(
        self,
        spatial_integrals: np.ndarray
    ) -> np.ndarray:
        """Converts spatial orbital integrals to spin-orbital basis."""
        if len(spatial_integrals.shape) == 2:
            # One-body integrals
            n = spatial_integrals.shape[0]
            spin_integrals = np.zeros((2*n, 2*n))
            
            # Map spatial to spin orbitals
            for p in range(n):
                for q in range(n):
                    spin_integrals[2*p, 2*q] = spatial_integrals[p, q]  # alpha-alpha
                    spin_integrals[2*p+1, 2*q+1] = spatial_integrals[p, q]  # beta-beta
                    
        elif len(spatial_integrals.shape) == 4:
            # Two-body integrals
            n = spatial_integrals.shape[0]
            spin_integrals = np.zeros((2*n, 2*n, 2*n, 2*n))
            
            # Map spatial to spin orbitals with proper antisymmetry
            for p in range(n):
                for q in range(n):
                    for r in range(n):
                        for s in range(n):
                            value = spatial_integrals[p, q, r, s]
                            # Same spin contributions
                            spin_integrals[2*p, 2*q, 2*r, 2*s] = value  # all alpha
                            spin_integrals[2*p+1, 2*q+1, 2*r+1, 2*s+1] = value  # all beta
                            
        return spin_integrals
    
    def _generate_fermionic_hamiltonian(
        self,
        one_body: np.ndarray,
        two_body: np.ndarray
    ) -> List[Tuple[float, List[Tuple[int, int]]]]:
        """Generates fermionic operator terms from integrals."""
        terms = []
        n_spin_orbitals = one_body.shape[0]
        
        # One-body terms
        for p in range(n_spin_orbitals):
            for q in range(n_spin_orbitals):
                if abs(one_body[p, q]) > 1e-12:
                    terms.append((
                        one_body[p, q],
                        [(p, 1), (q, 0)]  # creation, annihilation
                    ))
                    
        # Two-body terms
        for p in range(n_spin_orbitals):
            for q in range(n_spin_orbitals):
                for r in range(n_spin_orbitals):
                    for s in range(n_spin_orbitals):
                        if abs(two_body[p, q, r, s]) > 1e-12:
                            terms.append((
                                0.5 * two_body[p, q, r, s],
                                [(p, 1), (q, 1), (s, 0), (r, 0)]
                            ))
                            
        return terms
    
    def _jordan_wigner_transform(
        self,
        fermionic_terms: List[Tuple[float, List[Tuple[int, int]]]]
    ) -> SparsePauliOp:
        """Applies Jordan-Wigner transformation to fermionic operators."""
        pauli_strings = []
        coefficients = []
        
        for coefficient, term in fermionic_terms:
            # Generate Pauli string for this term
            pauli_string = self._term_to_pauli_string(term)
            pauli_strings.append(pauli_string)
            coefficients.append(coefficient)
            
        # Combine into SparsePauliOp
        return SparsePauliOp(pauli_strings, coeffs=coefficients)
    
    def _term_to_pauli_string(
        self,
        term: List[Tuple[int, int]]
    ) -> str:
        """Converts a fermionic term to Pauli string using Jordan-Wigner."""
        n_qubits = 2 * self.n_orbitals
        pauli_string = ['I'] * n_qubits
        
        for orbital, type in sorted(term, reverse=True):
            # type: 1 for creation, 0 for annihilation
            if type == 1:  # creation
                # Apply σ+ = (X-iY)/2
                for i in range(orbital):
                    pauli_string[i] = 'Z'
                pauli_string[orbital] = 'X'
            else:  # annihilation
                # Apply σ- = (X+iY)/2
                for i in range(orbital):
                    pauli_string[i] = 'Z'
                pauli_string[orbital] = 'X'
                
        return ''.join(pauli_string)
    
    def _combine_pauli_terms(
        self,
        pauli_terms: List[Tuple[float, str]]
    ) -> SparsePauliOp:
        """Combines Pauli terms into a SparsePauliOp."""
        pauli_strings = []
        coefficients = []
        for coeff, pauli_str in pauli_terms:
            pauli_strings.append(pauli_str)
            coefficients.append(coeff)
        return SparsePauliOp(pauli_strings, coefficients=coefficients)
    
    def _apply_symmetry_reduction(
        self,
        hamiltonian: SparsePauliOp
    ) -> SparsePauliOp:
        """Applies symmetry-based reduction to the Hamiltonian."""
        # Implement symmetry reduction based on:
        # - Particle number conservation
        # - Spin conservation
        # - Point group symmetries
        return self._reduce_by_symmetries(hamiltonian)
    
    def _reduce_by_symmetries(
        self,
        hamiltonian: SparsePauliOp
    ) -> SparsePauliOp:
        """Implements specific symmetry-based reductions."""
        # Start with particle number conservation
        reduced = self._apply_number_conservation(hamiltonian)
        
        # Apply spin conservation if applicable
        reduced = self._apply_spin_conservation(reduced)
        
        return reduced
    
    def _apply_number_conservation(
        self,
        hamiltonian: SparsePauliOp
    ) -> SparsePauliOp:
        """Applies particle number conservation symmetry."""
        # Filter terms that conserve particle number
        conserving_paulis = []
        conserving_coeffs = []
        
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            if self._conserves_particles(pauli):
                conserving_paulis.append(pauli)
                conserving_coeffs.append(coeff)
        
        # If no terms found, add identity term
        if not conserving_paulis:
            n_qubits = 2 * self.n_orbitals
            conserving_paulis.append('I' * n_qubits)
            conserving_coeffs.append(0.0)
                
        return SparsePauliOp(conserving_paulis, coeffs=conserving_coeffs)
    
    def _apply_spin_conservation(
        self,
        hamiltonian: SparsePauliOp
    ) -> SparsePauliOp:
        """Applies spin conservation symmetry."""
        # Filter terms that conserve total spin
        conserving_paulis = []
        conserving_coeffs = []
        
        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            if self._conserves_spin(pauli):
                conserving_paulis.append(pauli)
                conserving_coeffs.append(coeff)
        
        # If no terms found, add identity term
        if not conserving_paulis:
            n_qubits = 2 * self.n_orbitals
            conserving_paulis.append('I' * n_qubits)
            conserving_coeffs.append(0.0)
                
        return SparsePauliOp(conserving_paulis, coeffs=conserving_coeffs)
    
    def _conserves_particles(self, pauli_op) -> bool:
        """Checks if a term conserves particle number."""
        if isinstance(pauli_op, SparsePauliOp):
            # Handle SparsePauliOp
            for pauli_str in pauli_op.paulis:
                creators = sum(1 for op in pauli_str if op == 'X')
                if creators % 2 != 0:  # Must be even number for particle conservation
                    return False
            return True
        else:
            # Handle individual Pauli operator
            label = pauli_op.to_label()
            creators = sum(1 for op in label if op == 'X')
            return creators % 2 == 0  # Must be even number for particle conservation
    
    def _conserves_spin(self, pauli_op) -> bool:
        """Checks if a term conserves total spin."""
        if isinstance(pauli_op, SparsePauliOp):
            # Handle SparsePauliOp
            for pauli_str in pauli_op.paulis:
                spin_up = 0
                spin_down = 0
                
                # Count spin-up and spin-down operations
                for i, op in enumerate(pauli_str):
                    if op == 'X':  # X represents creation/annihilation
                        if i % 2 == 0:  # even indices are spin-up
                            spin_up += 1
                        else:  # odd indices are spin-down
                            spin_down += 1
                
                if spin_up != spin_down:  # Must conserve spin
                    return False
            return True
        else:
            # Handle individual Pauli operator
            label = pauli_op.to_label()
            spin_up = sum(1 for i, op in enumerate(label) 
                         if op == 'X' and i % 2 == 0)
            spin_down = sum(1 for i, op in enumerate(label) 
                          if op == 'X' and i % 2 == 1)
            return spin_up == spin_down  # Must conserve spin

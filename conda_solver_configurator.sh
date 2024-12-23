#!/bin/bash

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)

# Define conda solver configurations for different Ubuntu versions
declare -A CONDA_SOLVERS
CONDA_SOLVERS=(
  ["20.04"]="libmamba"  # Default for Ubuntu 20.04
  ["22.04"]="libmamba"  # Default for Ubuntu 22.04
  ["24.04"]="classic"   # Default for Ubuntu 24.04 (and future versions)
)

# Select conda solver based on Ubuntu version
CONDA_SOLVER=${CONDA_SOLVERS[$UBUNTU_VERSION]}
if [ -z "$CONDA_SOLVER" ]; then
  # Fallback to 'classic' solver for unknown Ubuntu versions
  CONDA_SOLVER="classic"
fi

# Configure conda to use the selected solver
conda config --set solver $CONDA_SOLVER

# Verify conda solver configuration
echo "Conda Solver Configuration:"
conda info --solver

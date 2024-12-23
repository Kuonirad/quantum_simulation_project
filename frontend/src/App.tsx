import React, { useState } from 'react';
import './App.css';

interface SimulationResult {
  energy: number;
  wavefunction: number[];
}

function App() {
  const [particles, setParticles] = useState<number>(1);
  const [potential, setPotential] = useState<string>('harmonic');
  const [result, setResult] = useState<SimulationResult | null>(null);

  const runSimulation = async () => {
    try {
      const response = await fetch('http://localhost:8000/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ particles, potential }),
      });
      const data: SimulationResult = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error running simulation:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Quantum Simulation App</h1>
        <div>
          <label>
            Number of Particles:
            <input type="number" value={particles} onChange={(e) => setParticles(Number(e.target.value))} min="1" />
          </label>
        </div>
        <div>
          <label>
            Potential Type:
            <select value={potential} onChange={(e) => setPotential(e.target.value)}>
              <option value="harmonic">Harmonic</option>
              <option value="square_well">Square Well</option>
            </select>
          </label>
        </div>
        <button onClick={runSimulation}>Run Simulation</button>
        {result && (
          <div>
            <h2>Simulation Results</h2>
            <p>Energy: {result.energy}</p>
            <p>Wavefunction: {result.wavefunction.join(', ')}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;

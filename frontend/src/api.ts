// src/api.ts
import { API_BASE } from "./config";

export interface SimulationRequest {
  d: number;
  hamiltonian: "diagonal_quadratic";
  initial_state: "basis" | "equal_superposition";
  basis_index?: number;
  t_max?: number;
  n_steps?: number;
}

export interface SimulationResponse {
  d: number;
  ts: number[];
  W: number[][][]; // [n_steps][d][d]
}

export async function runSimulation(
  params: SimulationRequest
): Promise<SimulationResponse> {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Simulation failed: ${res.status} ${msg}`);
  }

  return res.json();
}

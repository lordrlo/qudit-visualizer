import { API_BASE } from "./config";

export type InitialStateType = "basis" | "equal_superposition" | "custom";

export interface ComplexNumber {
  re: number;
  im: number;
}

export interface SimulationRequest {
  d: number;
  hamiltonian: "diagonal_quadratic";
  initial_state: InitialStateType;
  basis_index?: number;
  t_max?: number;
  n_steps?: number;

  // only used when initial_state === "custom"
  psi_custom?: ComplexNumber[];
}

export interface SimulationResponse {
  d: number;
  ts: number[];
  W: number[][][];
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

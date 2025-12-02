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
  W: number[][][];          // [n][d][d]
  psi: ComplexNumber[][];   // [n][d]
}


export type GateName = "X" | "Y" | "Z" | "F" | "T";

export interface GateRequest {
  d: number;
  gate: GateName;
  psi: ComplexNumber[];
}

export interface GateResponse {
  d: number;
  psi: ComplexNumber[];   // single state
  W: number[][];          // [d][d]
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

export async function applyGate(
  params: GateRequest
): Promise<GateResponse> {
  const res = await fetch(`${API_BASE}/apply_gate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Gate application failed: ${res.status} ${msg}`);
  }

  return res.json();
}

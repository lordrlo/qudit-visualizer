export interface ComplexNumber {
  re: number;
  im: number;
}

// initial state types used only on the frontend
export type InitialStateType =
  | "basis"
  | "equal_superposition"
  | "coherent"
  | "custom";

export type GateName = "X" | "Y" | "Z" | "F" | "T" | "custom";

export interface SimulationRequest {
  d: number;
  hamiltonian: string;
  initial_state: "custom"; // we always send custom, using psi_custom
  basis_index: number;
  t_max: number;
  n_steps: number;
  psi_custom: ComplexNumber[];
}

export interface SimulationResponse {
  d: number;
  ts: number[];          // length n_steps
  W: number[][][];       // [n_steps][d][d]
  psi: ComplexNumber[][]; // [n_steps][d]
}

export interface GateRequest {
  d: number;
  gate: GateName;
  psi: ComplexNumber[];
  U?: ComplexNumber[][]; // only used for gate === "custom"
}

export interface GateResponse {
  d: number;
  psi: ComplexNumber[]; // new state
  W: number[][];        // Wigner of new state [d][d]
}

export async function runSimulation(
  req: SimulationRequest
): Promise<SimulationResponse> {
  const resp = await fetch("http://127.0.0.1:8000/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Simulation failed: ${resp.status} ${text}`);
  }
  return resp.json();
}

export async function applyGate(
  req: GateRequest
): Promise<GateResponse> {
  const resp = await fetch("http://127.0.0.1:8000/apply_gate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Gate application failed: ${resp.status} ${text}`);
  }
  return resp.json();
}

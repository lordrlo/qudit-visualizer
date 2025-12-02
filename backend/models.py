from typing import List, Optional
from pydantic import BaseModel

class ComplexNumber(BaseModel):
    re: float
    im: float

class SimulationRequest(BaseModel):
    d: int
    hamiltonian: str  # "diagonal_quadratic" or "custom"
    initial_state: str
    basis_index: int
    t_max: float
    n_steps: int
    psi_custom: List[ComplexNumber]
    H_custom: Optional[List[List[ComplexNumber]]] = None  # NEW

class SimulationResponse(BaseModel):
    d: int
    ts: List[float]
    W: List[List[List[float]]]
    psi: List[List[ComplexNumber]]

class GateRequest(BaseModel):
    d: int
    gate: str
    psi: List[ComplexNumber]
    U: Optional[List[List[ComplexNumber]]] = None

class GateResponse(BaseModel):
    d: int
    psi: List[ComplexNumber]
    W: List[List[float]]

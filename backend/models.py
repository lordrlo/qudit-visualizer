# backend/models.py
from pydantic import BaseModel
from typing import Literal, List, Optional

InitialStateType = Literal["basis", "equal_superposition"]
HamiltonianType = Literal["diagonal_quadratic"]  # extend later

class SimulationRequest(BaseModel):
    d: int
    hamiltonian: HamiltonianType
    initial_state: InitialStateType
    basis_index: Optional[int] = 0
    t_max: float = 10.0
    n_steps: int = 201

class SimulationResponse(BaseModel):
    d: int
    ts: List[float]
    W: List[List[List[float]]]

from pydantic import BaseModel
from typing import Literal, List, Optional

InitialStateType = Literal["basis", "equal_superposition", "custom"]
HamiltonianType = Literal["diagonal_quadratic"]  # H stays fixed for now

class ComplexNumber(BaseModel):
    re: float
    im: float

class SimulationRequest(BaseModel):
    d: int
    hamiltonian: HamiltonianType
    initial_state: InitialStateType
    basis_index: Optional[int] = 0
    t_max: float = 10.0
    n_steps: int = 201

    # only used when initial_state == "custom"
    psi_custom: Optional[List[ComplexNumber]] = None

class SimulationResponse(BaseModel):
    d: int
    ts: List[float]
    W: List[List[List[float]]]

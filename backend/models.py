from pydantic import BaseModel
from typing import Literal, List, Optional

GateName = Literal["X", "Y", "Z", "F", "T"]

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
    W: List[List[List[float]]]     # [n_steps][d][d]
    psi: List[List[ComplexNumber]] # [n_steps][d]

from typing import List, Optional
from pydantic import BaseModel

class ComplexNumber(BaseModel):
    re: float
    im: float

class GateRequest(BaseModel):
    d: int
    gate: str                    # "X","Y","Z","F","T","custom"
    psi: List[ComplexNumber]     # input state
    U: Optional[List[List[ComplexNumber]]] = None  # only for gate == "custom"

class GateResponse(BaseModel):
    d: int
    psi: List[ComplexNumber]
    W: List[List[float]]
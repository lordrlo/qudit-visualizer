# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import dynamiqs as dq
import jax.numpy as jnp

from models import SimulationRequest, SimulationResponse
from wigner import phase_point_ops, wigner_from_rho

app = FastAPI(title="Discrete Wigner Simulator")

# Allow your frontend domain; for dev just use "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod if you like
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def build_hamiltonian(d: int, kind: str):
    if kind == "diagonal_quadratic":
        energies = jnp.array([(k * k) / d for k in range(d)], dtype=float)
        H = jnp.diag(energies)
        return H, energies
    else:
        raise ValueError(f"Unknown Hamiltonian type: {kind}")

def build_initial_state(
    d: int,
    initial_type: str,
    basis_index: int,
    psi_custom=None,
):
    if initial_type == "basis":
        idx = basis_index % d
        return dq.fock(d, idx)  # |idx>

    elif initial_type == "equal_superposition":
        # |psi> = (1/√d) Σ_k |k>
        vec = jnp.ones((d, 1), dtype=jnp.complex64) / jnp.sqrt(d)
        return vec   # sesolve accepts qarray-like

    elif initial_type == "custom":
        # psi_custom is a list of {"re": ..., "im": ...}
        if psi_custom is None or len(psi_custom) != d:
            raise ValueError("psi_custom must be a list of length d for custom state.")

        arr = jnp.array(
            [c.re + 1j * c.im for c in psi_custom],
            dtype=jnp.complex64,
        ).reshape(d, 1)

        # normalize so user doesn't have to get normalization perfect
        norm = jnp.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Custom state has zero norm.")
        return arr / norm

    else:
        raise ValueError(f"Unknown initial state type: {initial_type}")


@app.post("/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest):
    d = req.d
    if d % 2 == 0:
        raise HTTPException(status_code=400, detail="Only odd d supported currently.")

    # Build H and initial state
    H, _energies = build_hamiltonian(d, req.hamiltonian)
    try:
        psi0 = build_initial_state(
            d,
            req.initial_state,
            req.basis_index or 0,
            req.psi_custom,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


    # Time grid
    tsave = jnp.linspace(0.0, req.t_max, req.n_steps)

    # Solve Schrodinger equation
    result = dq.sesolve(H, psi0, tsave)
    states = result.states  # shape (n_steps, d, 1)

    # Phase-point operators
    A = phase_point_ops(d)

    W_list = []
    for n in range(req.n_steps):
        psi_t = states[n, :, 0]  # (d,)
        rho_t = jnp.outer(psi_t, jnp.conj(psi_t))  # (d,d)
        W = wigner_from_rho(rho_t, A)             # (d,d)
        W_list.append(W.tolist())

    response = SimulationResponse(
        d=d,
        ts=[float(t) for t in tsave],
        W=W_list,
    )
    return response

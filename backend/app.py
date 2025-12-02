# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import dynamiqs as dq
import jax.numpy as jnp

from models import (
    SimulationRequest,
    SimulationResponse,
    GateRequest,
    GateResponse,
    ComplexNumber,
)
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


def apply_gate_to_psi(psi: jnp.ndarray, gate: str, d: int) -> jnp.ndarray:
    """
    psi: column vector shape (d, 1)
    gate: "X", "Y", "Z", "F", "T"
    returns U_gate psi, same shape
    """
    omega = jnp.exp(2j * jnp.pi / d)
    idx = jnp.arange(d)

    if gate == "X":
        # cyclic shift: |q> -> |q+1>
        return jnp.roll(psi, shift=1, axis=0)

    elif gate == "Z":
        phases = omega ** idx  # shape (d,)
        return phases.reshape(d, 1) * psi

    elif gate == "Y":
        # define Y = Z X
        psi_x = jnp.roll(psi, shift=1, axis=0)
        phases = omega ** idx
        return phases.reshape(d, 1) * psi_x

    elif gate == "F":
        # discrete Fourier: F[p, q] = omega^(p q) / sqrt(d)
        p = idx.reshape(d, 1)
        q = idx.reshape(1, d)
        F = omega ** (p * q) / jnp.sqrt(d)
        return F @ psi

    elif gate == "T":
        # simple quadratic phase
        phases = jnp.exp(1j * jnp.pi * (idx ** 2) / d)
        return phases.reshape(d, 1) * psi

    else:
        raise ValueError(f"Unknown gate: {gate}")


@app.post("/simulate", response_model=SimulationResponse)
def simulate(req: SimulationRequest):
    d = req.d
    if d % 2 == 0:
        raise HTTPException(status_code=400, detail="Only odd d supported currently.")

    # Build H
    if req.hamiltonian == "custom":
        if req.H_custom is None:
            raise HTTPException(
                status_code=400,
                detail="Custom Hamiltonian requires H_custom matrix."
            )
        if len(req.H_custom) != d or any(len(row) != d for row in req.H_custom):
            raise HTTPException(
                status_code=400,
                detail="H_custom must be a d×d matrix."
            )

        H = jnp.array(
            [
                [c.re + 1j * c.im for c in row]
                for row in req.H_custom
            ],
            dtype=jnp.complex64,
        )

        # Optionally enforce Hermiticity softly:
        H = 0.5 * (H + jnp.conjugate(H.T))

    else:
        H, _energies = build_hamiltonian(d, req.hamiltonian)
        # ensure complex dtype
        H = jnp.array(H, dtype=jnp.complex64)

    # Build initial state
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

    psi_list = []
    W_list = []

    A = phase_point_ops(d)

    for n in range(req.n_steps):
        psi_t = states[n, :, 0]               # (d,)
        rho_t = jnp.outer(psi_t, jnp.conj(psi_t))
        W = wigner_from_rho(rho_t, A)        # (d,d)
        W_list.append(W.tolist())

        psi_list.append([
            {
                "re": float(jnp.real(z)),
                "im": float(jnp.imag(z)),
            }
            for z in psi_t
        ])

    return SimulationResponse(
        d=d,
        ts=[float(t) for t in tsave],
        W=W_list,
        psi=psi_list,
    )


@app.post("/apply_gate", response_model=GateResponse)
def apply_gate(req: GateRequest):
    d = req.d
    if d % 2 == 0:
        raise HTTPException(status_code=400, detail="Only odd d supported currently.")

    if len(req.psi) != d:
        raise HTTPException(status_code=400, detail="psi must have length d.")

    # build column vector from list of complex numbers
    psi = jnp.array(
        [c.re + 1j * c.im for c in req.psi],
        dtype=jnp.complex64,
    ).reshape(d, 1)

    # normalize to avoid drift
    norm = jnp.linalg.norm(psi)
    if norm == 0:
        raise HTTPException(status_code=400, detail="Input state has zero norm.")
    psi = psi / norm

    # apply gate: preset vs custom
    if req.gate == "custom":
        # --- custom unitary path ---
        if req.U is None:
            raise HTTPException(status_code=400, detail="Custom gate requires U.")

        if len(req.U) != d or any(len(row) != d for row in req.U):
            raise HTTPException(
                status_code=400,
                detail="Custom gate U must be a d×d matrix.",
            )

        # build JAX complex matrix from U
        U = jnp.array(
            [
                [u.re + 1j * u.im for u in row]
                for row in req.U
            ],
            dtype=jnp.complex64,
        )

        psi_new = U @ psi

    else:
        # --- preset gates: X, Y, Z, F, T ---
        try:
            psi_new = apply_gate_to_psi(psi, req.gate, d)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # normalize output (in case U is not exactly unitary)
    norm_new = jnp.linalg.norm(psi_new)
    if norm_new == 0:
        raise HTTPException(status_code=400, detail="Output state has zero norm.")
    psi_new = psi_new / norm_new

    # compute Wigner for the new state
    A = phase_point_ops(d)
    rho = psi_new @ jnp.conjugate(psi_new.T)
    W = wigner_from_rho(rho, A)  # shape (d, d)

    # convert psi_new back to list[ComplexNumber]
    psi_list: list[dict] = []
    psi_flat = psi_new[:, 0]
    for z in psi_flat:
        psi_list.append(
            {
                "re": float(jnp.real(z)),
                "im": float(jnp.imag(z)),
            }
        )

    return GateResponse(
        d=d,
        psi=psi_list,
        W=W.tolist(),
    )

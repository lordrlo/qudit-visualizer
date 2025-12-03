# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import dynamiqs as dq
import jax.numpy as jnp
import jax
from functools import lru_cache

from qudit_visualizer.models import (
    SimulationRequest,
    SimulationResponse,
    GateRequest,
    GateResponse,
    ComplexNumber,
)
from qudit_visualizer.wigner import phase_point_ops

app = FastAPI(title="Discrete Wigner Simulator")

# CORS: allow local dev + GitHub Pages frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://lordrlo.github.io",
    "https://lordrlo.github.io/qudit-visualizer/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,  # we don't need cookies/auth
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static frontend ---

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "static"

# Serve JS/CSS/assets under /qudit-visualizer/
app.mount("/qudit-visualizer", StaticFiles(directory=FRONTEND_DIST), name="frontend")

# Serve index.html at root
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIST / "index.html")
# Also serve index at /qudit-visualizer/
@app.get("/qudit-visualizer/")
def index_qudit():
    return FileResponse(FRONTEND_DIST / "index.html")


@lru_cache(maxsize=16)
def get_phase_point_ops(d: int):
    """Cached phase-point operators A(q,p) for given dimension d."""
    print(f"[backend] building phase-point operators for d={d}")
    return phase_point_ops(d)


@lru_cache(maxsize=16)
def _build_diagonal_quadratic(d: int):
    """Cached diagonal quadratic Hamiltonian and its eigenvalues."""
    energies = jnp.array([(k * k) / d for k in range(d)], dtype=float)
    H = jnp.diag(energies)
    return H, energies


@jax.jit
def wigner_from_psi_jit(psi, A):
    """Compute Wigner function from state vector |psi> and phase-point ops A."""
    rho = jnp.outer(psi, jnp.conj(psi))
    d = psi.shape[0]
    W = jnp.einsum("ij,qpji->qp", rho, A) / d
    return jnp.real(W)


def build_hamiltonian(d: int, kind: str):
    if kind == "diagonal_quadratic":
        return _build_diagonal_quadratic(d)
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
        # phase: |q> -> ω^q |q>
        phases = omega ** idx
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

    # --- Build H ---
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
        # softly enforce Hermiticity
        H = 0.5 * (H + jnp.conjugate(H.T))
    else:
        H, _energies = build_hamiltonian(d, req.hamiltonian)
        H = jnp.array(H, dtype=jnp.complex64)

    # --- Build initial state ---
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

    # Solve Schrödinger equation
    result = dq.sesolve(H, psi0, tsave)

    # result.states may be:
    # - a QArray (new dynamiqs) -> need dq.to_jax
    # - already an array-like (old dynamiqs)
    states_raw = result.states

    if hasattr(dq, "to_jax"):
        try:
            states = dq.to_jax(states_raw)
        except TypeError:
            # if states_raw is already an ndarray in some versions
            states = jnp.array(states_raw)
    else:
        states = jnp.array(states_raw)

    # states is now a jnp.ndarray, either shape (n_steps, d, 1) or (n_steps, d)
    psi_list = []
    W_list = []

    A = get_phase_point_ops(d)

    n_steps_eff = states.shape[0]

    for n in range(n_steps_eff):
        if states.ndim == 3:
            # (n_steps, d, 1)
            psi_t = states[n, :, 0]
        elif states.ndim == 2:
            # (n_steps, d)
            psi_t = states[n, :]
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected states array shape: {states.shape}",
            )

        W = wigner_from_psi_jit(psi_t, A)
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
    A = get_phase_point_ops(d)
    psi_flat = psi_new[:, 0]
    W = wigner_from_psi_jit(psi_flat, A)  # shape (d, d)

    psi_list = []
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


@app.on_event("startup")
def warmup() -> None:
    """Warm up JAX/dynamiqs and Wigner cache so the first real request is faster."""
    try:
        print("[startup] running warmup simulation...")
        d = 3
        dummy_req = SimulationRequest(
            d=d,
            hamiltonian="diagonal_quadratic",
            initial_state="basis",
            basis_index=0,
            t_max=1.0,
            n_steps=10,
            psi_custom=[ComplexNumber(re=0.0, im=0.0) for _ in range(d)],
            H_custom=None,
        )
        _ = simulate(dummy_req)
        print("[startup] warmup finished.")
    except Exception as e:
        # Warmup failures should not crash the app; just log them.
        print(f"[startup] warmup failed: {e!r}")

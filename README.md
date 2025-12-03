# Qudit Visualizer – Discrete Wigner Function

This project visualizes the discrete Wigner function of a single qudit of arbitrary dimension $d \ge 2$.  
It provides an interactive way to explore arbitrary states and dynamics in discrete phase space.

## Online demo

A live demo is available here:

👉 **https://lordrlo.github.io/qudit-visualizer/**

> **Performance note:** the online demo talks to a small remote backend and can be very slow for larger $d$ or long evolutions. For serious use (or just a smooth experience), you should run it locally.

You can: 

- Define arbitrary initial states of any finite dimension
- Evolve them with arbitrary gates and Hamiltonians
- Inspect the resulting Wigner function $W(q,p,t)$

---

## Quick start

Install from PyPI with

```bash
pip install qudit-visualizer
qudit-visualizer
```

and enjoy! The website will open at http://localhost:8000. 

## Features

### Qudit and Wigner function

- Single qudit with dimension $d \ge 2$ (odd or even).
- Discrete phase space with coordinates $(q,p)$ in $\mathbb{Z}_d \times \mathbb{Z}_d$.
- Wigner function computed as 

  $$W(q,p) = \frac{1}{d}\mathrm{Tr}\big[\rho A_{q,p}\big]$$

  where $\rho = \ket{\psi}\bra{\psi}$ and $A_{q,p}$ are phase–point operators (see the theory section for details).

### Initial states

Initial states are configured on the frontend and sent to the backend as a normalized vector $\lvert\psi\rangle$. Supported presets:

- **Computational basis state** $\lvert i\rangle$  
  with a slider for the index $i$.
- **Equal superposition**

  $$\lvert\psi\rangle = \frac{1}{\sqrt{d}} \sum_{k=0}^{d-1} \lvert k\rangle$$

- **“Coherent”-like state** $\lvert q,p\rangle$  
  constructed via a discrete displacement $D_{q,p}\lvert 0\rangle$, with tunable $(q,p)$.
- **Custom state**  
  - Specify amplitudes $\psi_i$ as mathjs expressions, e.g.
    `exp(i*pi/3)/sqrt(2)`, `cos(2*pi) + 3*i`.  
  - The backend normalizes the state when you press **Generate**.

### Evolution modes

Two complementary evolution modes are available.

#### 1. Gate mode

Apply a single unitary gate to the current state:

- **Built-in gates**:
  - **$X$** – cyclic shift in the computational basis,
  - **$Z$** – phase gate $Z\lvert q\rangle = \omega^q \lvert q\rangle$,
  - **$F$** – discrete Fourier transform,
  - **$T$** – simple quadratic phase gate.
- **Custom gate**:
  - Choose any $d \times d$ unitary matrix that will be used as a gate.

After applying the gate, the backend:

1. Normalizes the resulting state,
2. Computes the new Wigner function $W(q,p)$,
3. Returns the updated state and Wigner function.

#### 2. Continuous evolution mode

Solve the time-dependent Schrödinger equation

$$
i \frac{d}{dt}\lvert\psi(t)\rangle = H \lvert\psi(t)\rangle
$$

using Dynamiqs on top of JAX.

- Time parameters:
  - Final time $t_{\max}$,
- Hamiltonian choices:
  - **Preset**: diagonal quadratic spectrum $H_{i,i} = i^2 / d$,
  - **Custom**: full $d \times d$ Hermitian matrix specified in the UI.

For each saved time point, the backend:

1. Extracts $\lvert\psi(t_n)\rangle$,
2. Builds $\rho(t_n) = \lvert\psi(t_n)\rangle\langle\psi(t_n)\rvert$,
3. Computes $W(q,p;t_n)$,
4. Returns the time series $\{t_n\}$, states, and Wigner functions.

### Visualization & controls

- Wigner function displayed as a heatmap on a canvas:
  - **blue** for negative values,
  - **white** near zero,
  - **red** for positive values.
- Hover tooltips show $W(q,p)$ at the cursor.
- For continuous evolution:
  - Playback controls (play/pause),
  - Frame slider,
  - Speed control.

---

## Project structure

Repository layout (top level):

```text
.
├── qudit_visualizer/
│   ├── __init__.py          # Package init
│   ├── app.py               # FastAPI app: endpoints, dynamics, Wigner + static frontend
│   ├── models.py            # Pydantic models for requests/responses
│   ├── wigner.py            # Phase–point operators A(q,p) and Wigner map
│   └── static/              # Prebuilt frontend bundle for packaged use
├── frontend/
│   ├── .env.development     # VITE_API_BASE for local dev
│   ├── .env.production      # VITE_API_BASE for GitHub Pages demo
│   ├── .env.localapp        # VITE_API_BASE for packaged app (usually unset)
│   ├── index.html
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx              # Main React app: UI, controls, state management
│       ├── api.ts               # Typed API client for /simulate, /apply_gate
│       ├── config.ts            # API_BASE configuration
│       ├── components/
│       │   └── WignerHeatmap.tsx  # Canvas-based Wigner heatmap component
│       ├── App.css, index.css   # Styling
│       └── main.tsx             # React entry point
├── benchmark.py             # Local benchmark script for backend performance
├── pyproject.toml           # Packaging config for PyPI
└── README.md                # (This file)
```

---

## Tech stack

### Backend

- Python 3
- FastAPI – HTTP API server
- [Dynamiqs](https://github.com/dynamiqs/dynamiqs) – solvers for Schrödinger dynamics on top of JAX
- JAX / jax.numpy – linear algebra and fast array operations
- uvicorn – ASGI server for development and deployment

Main HTTP endpoints (see `qudit_visualizer/app.py`):

- `POST /simulate`  
  Runs continuous time evolution with chosen Hamiltonian and initial state.
- `POST /apply_gate`  
  Applies a single unitary gate and returns the updated state and Wigner function.

### Frontend

- React + TypeScript
- Vite – dev server and bundler
- mathjs – parsing and evaluating complex-valued expressions in the UI
- Plain CSS for layout and styling; plotting uses a simple HTML `<canvas>`

---

## Theory overview

1. **Generalized Pauli operators**  
   For a $d$-dimensional Hilbert space with computational basis $\{\lvert q\rangle : q \in \mathbb{Z}_d\}$, define

   - $X\lvert q\rangle = \lvert q + 1 \bmod d\rangle$,
   - $Z\lvert q\rangle = \omega^q \lvert q\rangle$, with $\omega = e^{2\pi i / d}$.

   These generate the discrete Weyl–Heisenberg group and underpin the phase–space structure.

2. **Phase space and phase–point operators**  
   The discrete phase space is

   $$\Gamma_d = \{(q,p)\}= \mathbb{Z}_d \times \mathbb{Z}_d.$$

   To each phase–space point $(q,p)$ we associate a Hermitian phase–point operator $A_{q,p}$.  
   The Wigner function is then

   $$W(q,p) = \frac{1}{d}\mathrm{Tr}\big[\rho A_{q,p}\big]$$

   and satisfies:

   - Real-valuedness,
   - Normalization $\sum_{q,p} W(q,p) = 1$,
   - Correct marginals (line sums reproduce measurement probabilities),
   - Possible negativity for nonclassical states.

3. **Odd vs even dimension**  

   - For odd $d$ there is a simple closed-form expression for $A_{q,p}$ in terms of shifts in the computational basis, which is implemented directly in the code (see the `phase_point_ops_odd` path in `wigner.py`). One convenient expression is

     $$A_{q,p} = \sum_{s=0}^{d-1} \omega^{2ps} \lvert q + s\rangle\langle q - s\rvert$$

     with indices understood modulo $d$.

   - For even $d$, constructing a Wigner function on a $d \times d$ lattice with good axiomatic properties is subtler. The implementation in this repository follows the stencil-based framework of:

     > Lucky K. Antonopoulos, Joseph F. Fitzsimons, Adam G. M. Lewis, and Antoine Tilloy,  
     > “Grand Unification of All Discrete Wigner Functions on $d \times d$ Phase Space”,  
     > PRA 112, 052219 (2025), https://doi.org/10.1103/s5wn-mysr

     In short:

     1. Build a “parent” Wigner kernel on a $2d \times 2d$ phase space using Weyl operators $X^m Z^n$ and a parity operator $P$.
     2. Construct parent phase–point operators $A^{(2d)}(m_1,m_2)$ on that larger lattice.
     3. Compress the result to a $d \times d$ phase space by averaging over a $2 \times 2$ stencil:

        $$A(q,p) = \frac{1}{2} \sum_{b_1,b_2 \in \{0,1\}}A^{(2d)}(2q + b_1, 2p + b_2).$$

     This guarantees a consistent Wigner function on the $d \times d$ lattice for even $d$.

4. **Dynamics**  

   - Gate mode: $\lvert\psi'\rangle = U \lvert\psi\rangle$ for a chosen or custom unitary $U$.
   - Continuous mode: integrate $i\, d\lvert\psi\rangle/dt = H\lvert\psi\rangle$ with Dynamiqs, sample
     $\lvert\psi(t_n)\rangle$, and compute $W(q,p;t_n)$ at each time.

---

### Optional: backend benchmark

To get a quick feeling for backend-only performance:

```bash
python benchmark.py --runs 5 --d 10 --steps 200 --tmax 10.0
```

This runs a few local simulations (without HTTP) and prints timing statistics for both `simulate` and `apply_gate`.

---

## Roadmap and ideas

Some natural extensions that this codebase is ready for:

- Multi-qudit phase spaces and Wigner functions,
- Open-system dynamics (Lindblad) with Dynamiqs,
- Saving/loading parameter presets and simulation results.

Contributions, bug reports, and feature requests are welcome!

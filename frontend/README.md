# Qudit Visualizer – Discrete Wigner Function in \(d\) Dimensions

This project visualizes the **discrete Wigner function** of a single qudit of arbitrary dimension \(d \ge 2\).  
It supports both **odd and even** local dimensions and provides an interactive way to explore states and dynamics directly in discrete phase space.

For **even \(d\)** the phase–point operators are built using the stencil-based framework of

> Lucky K. Antonopoulos, Joseph F. Fitzsimons, Adam G. M. Lewis, and Antoine Tilloy,  
> **“Grand Unification of All Discrete Wigner Functions on \(d \times d\) Phase Space”**,  
> arXiv:2503.09353 (2025), https://arxiv.org/pdf/2503.09353

which constructs “parent” Wigner kernels on a \(2d \times 2d\) lattice and compresses them to a \(d \times d\) phase space.

For **odd \(d\)**, a simpler Wootters-style construction is used.

You can:

- Choose or define **initial states**,
- Evolve them either by **discrete gates** or **continuous time evolution** under a Hamiltonian,
- And inspect the resulting **Wigner function \(W(q,p)\)** as a static plot (gate mode) or as an **animation** (continuous mode).

---

## Online demo

A live demo of the frontend is available here:

👉 **https://lordrlo.github.io/qudit-visualizer/**

> **Performance note:** the online demo talks to a small remote backend and can be **very slow** for larger \(d\) or long evolutions.  
> For serious use (or just a smooth experience), you should run **both backend and frontend locally**, as described in the installation section.

---

## Features

### Qudit and Wigner function

- Single qudit with **dimension \(d \ge 2\)** (odd or even).
- Discrete phase space with coordinates \((q,p) \in \{0,\dots,d-1\}^2\).
- Wigner function computed as
  \[
    W(q,p) = \frac{1}{d} \operatorname{Tr}[\rho\, A_{q,p}]
  \]
  where \(\rho = |\psi\rangle\langle\psi|\) and \(A_{q,p}\) are phase–point operators:
  - For **odd \(d\)**: Wootters-style kernel,
    \[
      A_{q,p} = \sum_{s=0}^{d-1} \omega^{2ps} \, |q+s\rangle\langle q-s|
      \quad\text{with}\quad
      \omega = e^{2\pi i / d}.
    \]
  - For **even \(d\)**: a **compressed \(2d \times 2d\)** construction following Antonopoulos *et al.*,
    starting from a parent kernel \(A^{(2d)}(m_1,m_2) = X^{m_1} Z^{m_2} P \exp(i\pi m_1 m_2 / d)\) and averaging:
    \[
      A(q,p) = \frac{1}{2} \sum_{b_1,b_2=0}^1
      A^{(2d)}(2q + b_1, 2p + b_2).
    \]

(Indices are understood modulo \(d\); see `backend/wigner.py` for the exact implementation.)

### Initial states

Initial states are configured on the frontend and sent to the backend as a normalized vector \(|\psi\rangle\). Supported presets:

- **Computational basis state** \(|i\rangle\)  
  with a slider for the index \(i\).
- **Equal superposition**  
  \[
    |\psi\rangle = \frac{1}{\sqrt{d}} \sum_{k=0}^{d-1} |k\rangle.
  \]
- **“Coherent”-like state** \(|q,p\rangle\)  
  constructed via a discrete displacement \(D_{q,p}|0\rangle\), with tunable \((q,p)\).
- **Custom state**  
  - Specify amplitudes \(\psi_i\) as **mathjs expressions**, e.g.
    `exp(i*pi/3)/sqrt(2)`, `cos(2pi) + 3i`.  
  - The backend normalizes the state when you press **Generate**.

### Evolution modes

Two complementary evolution modes are available.

#### 1. Gate mode

Apply a **single unitary gate** to the current state:

- **Built‑in gates**:
  - **X** – cyclic shift in the computational basis,
  - **Z** – phase gate \(Z|q\rangle = \omega^q |q\rangle\),
  - **F** – discrete Fourier transform,
  - **T** – simple quadratic phase gate.
- **Custom gate**:
  - Choose any \(d \times d\) unitary matrix.

After applying the gate, the backend:

1. Normalizes the resulting state,
2. Computes the new Wigner function \(W(q,p)\),
3. Returns the updated state and Wigner function.

#### 2. Continuous evolution mode

Solve the **time-dependent Schrödinger equation**

\[
i \frac{d}{dt}|\psi(t)\rangle = H |\psi(t)\rangle
\]

using [Dynamiqs](https://github.com/dynamiqs/dynamiqs) on top of JAX.

- Time parameters:
  - Final time \(t_{\max}\),
  - Number of time steps \(N\).
- Hamiltonian choices:
  - **Preset**: diagonal quadratic spectrum \(H_{kk} = k^2 / d\),
  - **Custom**: full \(d \times d\) complex matrix specified in the UI.

For each saved time point, the backend:

1. Extracts \(|\psi(t_n)\rangle\),
2. Builds \(\rho(t_n) = |\psi(t_n)\rangle\langle\psi(t_n)|\),
3. Computes \(W(q,p; t_n)\),
4. Returns the time series \(\{t_n\}\), states and Wigner functions.

### Visualization & controls

- Wigner function displayed as a **heatmap** on a canvas:
  - **blue** for negative values,
  - **white** near zero,
  - **red** for positive values.
- Hover tooltips show \(W(q,p)\) at the cursor.
- For continuous evolution:
  - Playback controls (play/pause),
  - Frame slider,
  - Speed control.

---

## Project structure

Repository layout (top level):

```text
.
├── backend/
│   ├── app.py          # FastAPI app: endpoints, dynamics, Wigner computation
│   ├── models.py       # Pydantic models for requests/responses
│   └── wigner.py       # Phase–point operators A(q,p) and Wigner map
├── frontend/
│   ├── .env.development    # VITE_API_BASE for local dev
│   ├── .env.production     # VITE_API_BASE for deployed demo
│   ├── index.html
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx             # Main React app: UI, controls, state management
│       ├── api.ts              # Typed API client for /simulate, /apply_gate
│       ├── config.ts           # API_BASE configuration
│       ├── components/
│       │   └── WignerHeatmap.tsx  # Canvas-based Wigner heatmap component
│       ├── App.css, index.css  # Styling
│       └── main.tsx            # React entry point
├── benchmark.py        # Local benchmark script for backend performance
├── requirements.txt    # Python backend dependencies
└── README.md           # (This file)
```

---

## Tech stack

### Backend

- **Python 3**
- **FastAPI** – HTTP API server.
- **Dynamiqs** – solvers for Schrödinger dynamics on top of JAX.
- **JAX / jax.numpy** – linear algebra and fast array operations.
- **uvicorn** – ASGI server for development and deployment.

Main HTTP endpoints (see `backend/app.py`):

- `POST /simulate`
  - Body: `SimulationRequest`
  - Runs continuous time evolution with chosen Hamiltonian and initial state.
- `POST /apply_gate`
  - Body: `GateRequest`
  - Applies a single unitary gate and returns the updated state & Wigner function.
- `GET /health`
  - Lightweight health check.

### Frontend

- **React** + **TypeScript**
- **Vite** – dev server and bundler.
- **mathjs** – parsing and evaluating complex-valued expressions in the UI.
- Plain CSS for layout and styling; plotting uses a simple HTML `<canvas>`.

---

## Short theory overview

1. **Generalized Pauli operators**  
   For a \(d\)-dimensional Hilbert space with computational basis \(\{|q\rangle\}_{q=0}^{d-1}\), define
   \[
   X|q\rangle = |q+1 \mod d\rangle, \qquad
   Z|q\rangle = \omega^q |q\rangle, \quad \omega = e^{2\pi i/d}.
   \]
   These generate the discrete Weyl–Heisenberg group and underpin the phase–space structure.

2. **Phase–space & phase–point operators**  
   The discrete Wigner function is defined on a finite phase space
   \[
   \Gamma_d = \{(q,p) : q,p \in \mathbb{Z}_d\}.
   \]
   To each phase–space point \((q,p)\) we associate a Hermitian **phase–point operator** \(A_{q,p}\).  
   The Wigner function is then
   \[
     W(q,p) = \frac{1}{d} \operatorname{Tr}[\rho A_{q,p}]
   \]
   and satisfies:
   - Real-valuedness,
   - Normalization \(\sum_{q,p} W(q,p) = 1\),
   - Correct marginals (line sums reproduce measurement probabilities),
   - Possible negativity for nonclassical states.

3. **Odd vs even dimension**  
   - For **odd \(d\)** there is a simple closed-form expression for \(A_{q,p}\) in terms of shifts in the computational basis, which is implemented directly in `phase_point_ops_odd`.
   - For **even \(d\)**, constructing a Wigner function on a \(d \times d\) lattice with good axiomatic properties is subtler.  
     Following Antonopoulos *et al.*, we:
     1. Build a “parent” Wigner kernel on a \(2d \times 2d\) phase space,
     2. Use a parity operator \(P\) and the Weyl operators \(X^m Z^n\),
     3. Compress the result to a \(d \times d\) phase space via a \(2 \times 2\) stencil.

4. **Dynamics**  
   - **Gate mode:** \( |\psi'\rangle = U|\psi\rangle\) for a chosen or custom unitary \(U\).
   - **Continuous mode:** integrate \(i\, d|\psi\rangle/dt = H|\psi\rangle\) with Dynamiqs, sample
     \(|\psi(t_n)\rangle\), and compute \(W(q,p; t_n)\) at each time.

---

## Installation & local use

> **Recommended:** For good performance, run the project locally instead of using the online demo.

### 1. Clone the repository

```bash
git clone https://github.com/lordrlo/qudit-visualizer.git
cd qudit-visualizer/qudit-visualizer-main
```

(Adjust the path/URL if you are working from a fork or a different layout.)

---

### 2. Backend setup (FastAPI + Dynamiqs)

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the backend with uvicorn:

```bash
uvicorn backend.app:app --reload
```

By default this starts the API at **http://localhost:8000**.

> If you want GPU acceleration, install a JAX/JAXLIB build compatible with your CUDA/ROCm stack before installing `dynamiqs`. The project itself does not require a GPU and works on CPU-only installs (just slower).

---

### 3. Frontend setup (React + Vite)

In a **separate terminal**, leaving the backend running:

```bash
cd frontend
npm install          # or pnpm/yarn if you prefer
npm run dev
```

This starts the Vite dev server at something like **http://localhost:5173**.

The file `frontend/.env.development` already sets

```bash
VITE_API_BASE=http://localhost:8000
```

so the frontend will automatically talk to your local backend.

Open the printed URL (usually `http://localhost:5173/`) in your browser.

---

### 4. Optional: production build

To build a static bundle of the frontend:

```bash
cd frontend
npm run build
npm run preview   # serves the built bundle locally
```

You can also deploy the contents of `frontend/dist/` to any static hosting service and set `VITE_API_BASE` to the URL of your backend (for example, on a VPS).

---

### 5. Optional: backend benchmark

To get a quick feeling for **backend-only** performance:

```bash
python benchmark.py --runs 5 --d 10 --steps 200 --tmax 10.0
```

This runs a few local simulations (without HTTP) and prints timing statistics for both `simulate` and `apply_gate`.

---

## Roadmap / ideas

Some natural extensions that this codebase is ready for:

- Multi‑qudit phase spaces and Wigner functions,
- Open‑system dynamics (Lindblad) with Dynamiqs,
- Saving/loading parameter presets and simulation results.

Contributions, bug reports, and feature requests are welcome!

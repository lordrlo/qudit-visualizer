# Discrete Wigner Visualizer

Interactive visualizer for the **discrete Wigner function** of a single qudit (finite-dimensional quantum system) evolving under a fixed Hamiltonian.

The project is split into:

- a **Python backend** (FastAPI + [Dynamiqs](https://github.com/dynamiqs/dynamiqs)) that solves the Schrödinger equation for a qudit,
- a **React + TypeScript frontend** that displays the discrete Wigner function `W(q, p; t)` as an animated heatmap.

This is meant as both an educational tool and a portfolio project showcasing numerical quantum dynamics and modern web dev.

---

## Features

Initial version (this repo) supports:

- Single **qudit** of dimension `d` (currently **odd** `d`: 3, 5, 7, …).
- Time evolution under a **fixed diagonal Hamiltonian**

  - Energies: `E_k = k^2 / d` for `k = 0, 1, ..., d-1`
  - Hamiltonian matrix: `H = diag(E_0, ..., E_{d-1})`

- Two initial state presets:

  - **Basis state** `|q0>` in the computational basis
  - **Equal superposition** `(1/sqrt(d)) * sum_q |q>`

- **Discrete Wigner function**

  - `W(q, p; t) = (1/d) * Tr[ rho(t) * A_{q,p} ]`
  - where `rho(t) = |psi(t)><psi(t)|` and `A_{q,p}` are phase-point operators (see Theory below).

- Animated **heatmap** of `W(q, p; t)`:

  - Red = positive quasi-probability
  - Blue = negative quasi-probability
  - Intensity = magnitude `|W(q, p; t)|`

- Interactive controls:

  - Dimension `d` (odd)
  - Initial state choice (`|q>` or equal superposition)
  - Time slider
  - Play / pause and speed control
  - **Hover** over a cell to see `(q, p, W(q, p))` numerically in real time.

---

## Tech stack

**Backend**

- Python 3
- [Dynamiqs](https://github.com/dynamiqs/dynamiqs) (JAX-based quantum dynamics)
- FastAPI
- Uvicorn

**Frontend**

- React + TypeScript
- Vite
- Plain CSS / inline styles for layout
- HTML canvas for the heatmap

---

## Theory overview (informal)

We consider a single qudit of dimension `d` with computational basis `{ |q> }` for `q = 0, ..., d-1`.

### Hamiltonian

In this initial version the Hamiltonian is fixed to a diagonal “discrete oscillator” style spectrum:

- `E_k = k^2 / d`
- `H = diag(E_0, ..., E_{d-1})`

The time-dependent state is

- `|psi(t)> = exp(-i H t) |psi(0)>`
- `rho(t) = |psi(t)><psi(t)|`

### Discrete Wigner function (odd d)

For **odd** `d`, we define phase–point operators

- `A_{q,p} = sum_s omega^(2 p s) |q + s><q - s|`
- with `omega = exp(2 pi i / d)` and all indices understood modulo `d`.

These operators satisfy:

- `Tr(A_{q,p}) = 1`
- `Tr(A_{q,p} A_{q',p'}) = d * delta_{q,q'} * delta_{p,p'}`

The **discrete Wigner function** is defined as

- `W(q, p; t) = (1/d) * Tr[ rho(t) * A_{q,p} ]`

Properties:

- `W(q, p; t)` is **real-valued**.
- It is **normalized**: sum over all `q, p` gives 1.
- It can be negative: it is a **quasi-probability distribution**, not a true probability distribution.

The visualization shows `W(q, p; t)` on a `d x d` grid as a diverging colormap:

- **red** = positive values,
- **blue** = negative values,
- **white** ≈ zero.

---

## Project structure

```text
qudit_visualizer/
  backend/
    app.py          # FastAPI app + Dynamiqs solver
    models.py       # Pydantic request/response models
    wigner.py       # phase-point operators + Wigner computation
    requirements.txt
  frontend/
    index.html
    package.json
    vite.config.ts
    src/
      main.tsx
      App.tsx
      api.ts        # calls the /simulate endpoint
      config.ts     # API_BASE configuration
      index.css
      components/
        WignerHeatmap.tsx
```

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/lordrlo/qudit_visualizer.git
cd qudit_visualizer
```

### 2. Backend setup (FastAPI + Dynamiqs)

Create and activate a virtual environment (optional but recommended):

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # on macOS / Linux
# .venv\Scripts\activate  # on Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn app:app --reload --port 8000
```

This starts a FastAPI server at `http://localhost:8000`.

### 3. Frontend setup (React + Vite)

In another terminal:

```bash
cd qudit_visualizer/frontend
npm install
```

If needed, configure the backend URL (defaults to `http://localhost:8000`):

```bash
# frontend/.env.local  (optional)
VITE_API_BASE=http://localhost:8000
```

Run the dev server:

```bash
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`).

---

## How to use

1. Start the backend:

   ```bash
   cd backend
   uvicorn app:app --reload --port 8000
   ```

2. Start the frontend:

   ```bash
   cd frontend
   npm run dev
   ```

3. Open the frontend in your browser.

4. In the left sidebar:

   - Choose the **dimension** `d` (3, 5, 7, …).
   - Select the **initial state**:
     - *Basis state*: use the slider to pick `|q0>`.
     - *Equal superposition*: the uniform superposition of all basis states.

5. Click **Run simulation**. The frontend sends a request to the backend, which:

   - solves the Schrödinger equation with Dynamiqs,
   - computes `W(q, p; t_n)` for each time step `t_n`,
   - returns the data as JSON.

6. Use:

   - the **time slider** to move through the evolution,
   - the **Play** button and **Speed** slider for continuous animation,
   - **hover** over any square to see `(q, p, W(q, p))` numerically.

---

## Limitations / future directions

Current version:

- Single qudit only.
- Dimension `d` must be **odd** (the phase–point operator construction assumes odd `d`).
- Hamiltonian is **fixed diagonal** with quadratic spectrum.
- Pure-state unitary evolution only (no open systems / noise).

Possible extensions:

- Custom initial states (user-specified complex amplitudes in the computational basis).
- Custom Hamiltonians (diagonal or fully general Hermitian matrices).
- Support for open-system dynamics via Dynamiqs `mesolve` (collapse operators).
- Alternative Wigner constructions for even `d` (e.g. qubits and qubit registers).
- Additional visualizations: probability distributions `|psi_q|^2`, marginals, Wigner negativity.



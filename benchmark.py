"""
Local benchmark script for the qudit-visualizer backend.

Example:
    python benchmark.py --runs 5 --d 5 --steps 200 --tmax 10.0
"""

import time
import statistics as stats
from typing import List

from backend.models import SimulationRequest, GateRequest, ComplexNumber
from backend.app import simulate, apply_gate


def make_sim_request(d: int, t_max: float, n_steps: int) -> SimulationRequest:
    """Build a SimulationRequest for a simple diagonal_quadratic Hamiltonian."""
    return SimulationRequest(
        d=d,
        hamiltonian="diagonal_quadratic",
        initial_state="basis",
        basis_index=0,
        t_max=t_max,
        n_steps=n_steps,
        psi_custom=[ComplexNumber(re=0.0, im=0.0) for _ in range(d)],
        H_custom=None,
    )


def make_gate_request(d: int) -> GateRequest:
    """Build a GateRequest using the Fourier gate F on |0>."""
    psi = [ComplexNumber(re=1.0 if i == 0 else 0.0, im=0.0) for i in range(d)]
    return GateRequest(
        d=d,
        gate="F",
        psi=psi,
        U=None,
    )


def time_call(fn, *args, **kwargs) -> float:
    """Time a single call to fn(*args, **kwargs) and return duration in seconds."""
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    t1 = time.perf_counter()
    return t1 - t0


def benchmark_simulate(runs: int, d: int, t_max: float, n_steps: int) -> None:
    req = make_sim_request(d, t_max, n_steps)
    times: List[float] = []

    # warm-up run (compile JAX etc.)
    simulate(req)

    for _ in range(runs):
        times.append(time_call(simulate, req))

    print(f"[simulate] d={d}, t_max={t_max}, n_steps={n_steps}, runs={runs}")
    print(f"  min   = {min(times):.4f} s")
    print(f"  max   = {max(times):.4f} s")
    print(f"  mean  = {stats.mean(times):.4f} s")
    if len(times) >= 2:
        print(f"  stdev = {stats.stdev(times):.4f} s")


def benchmark_apply_gate(runs: int, d: int) -> None:
    req = make_gate_request(d)
    times: List[float] = []

    # warm-up run
    apply_gate(req)

    for _ in range(runs):
        times.append(time_call(apply_gate, req))

    print(f"[apply_gate] d={d}, gate={req.gate}, runs={runs}")
    print(f"  min   = {min(times):.4f} s")
    print(f"  max   = {max(times):.4f} s")
    print(f"  mean  = {stats.mean(times):.4f} s")
    if len(times) >= 2:
        print(f"  stdev = {stats.stdev(times):.4f} s")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local backend benchmark (no HTTP).")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per benchmark.")
    parser.add_argument("--d", type=int, default=5, help="Qudit dimension.")
    parser.add_argument("--steps", type=int, default=200, help="Number of time steps for simulate.")
    parser.add_argument("--tmax", type=float, default=10.0, help="Maximum time for simulate.")

    args = parser.parse_args()

    print("=== Benchmark simulate ===")
    benchmark_simulate(args.runs, args.d, args.tmax, args.steps)
    print()
    print("=== Benchmark apply_gate ===")
    benchmark_apply_gate(args.runs, args.d)


if __name__ == "__main__":
    main()

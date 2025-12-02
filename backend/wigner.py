# backend/wigner.py
import jax.numpy as jnp

def phase_point_ops(d: int):
    """
    Build A_{q,p} = sum_s omega^{2 p s} |q+s><q-s|
    for odd d. Returns array A[q,p,i,j].
    """
    if d % 2 == 0:
        raise ValueError("This simple construction assumes odd d.")

    omega = jnp.exp(2j * jnp.pi / d)
    A = []
    for q in range(d):
        row = []
        for p in range(d):
            M = jnp.zeros((d, d), dtype=complex)
            for s in range(d):
                phase = omega ** (2 * p * s)
                ket = (q + s) % d
                bra = (q - s) % d
                M = M.at[ket, bra].add(phase)
            row.append(M)
        A.append(row)
    return jnp.array(A)  # shape (d, d, d, d)


def wigner_from_rho(rho, A):
    """
    W(q,p) = (1/d) Tr[rho A_{q,p}] for all q,p.
    rho: (d,d)
    A:   (d,d,d,d)
    returns W: (d,d), real
    """
    d = rho.shape[0]
    W = jnp.einsum("ij,qpji->qp", rho, A) / d
    return jnp.real(W)
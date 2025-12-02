# backend/wigner.py
import jax.numpy as jnp

def phase_point_ops(d: int) -> jnp.ndarray:
    """
    Unified dispatcher: works for any d >= 2.
    """
    if d <= 1:
        raise ValueError("d must be >= 2.")

    if d % 2 == 1:
        return phase_point_ops_odd(d)
    else:
        return phase_point_ops_even(d)
    
def phase_point_ops_even(d: int) -> jnp.ndarray:
    """
    Phase-point operators A(q,p) for even d using the
    Antonopoulos/Leonhardt-style 2d×2d parent construction
    compressed to a d×d lattice.

    Returns:
        A: array of shape (d, d, d, d), where
           A[q, p] is a (d, d) matrix (the phase-point operator).
    """
    if d % 2 != 0:
        raise ValueError("phase_point_ops_even called with odd d.")

    dtype = jnp.complex64
    dim = d

    omega = jnp.exp(2j * jnp.pi / dim)

    # --- Weyl pair X,Z ---

    # X|q> = |q+1>
    eye = jnp.eye(dim, dtype=dtype)
    X = jnp.roll(eye, shift=1, axis=0)  # permutation matrix

    # Z|q> = ω^q |q>
    q = jnp.arange(dim)
    Z = jnp.diag(omega ** q).astype(dtype)

    # --- Parity P|q> = |-q mod d>
    #
    # E = I_d; its columns are the standard basis vectors e_0, e_1, ...
    # We want column q of P to be e_{-q mod d}, so:
    #   P = E[:, perm] with perm[q] = (-q) mod d
    perm = (-q) % dim               # shape (d,)
    E = jnp.eye(dim, dtype=dtype)
    P = E[:, perm]                  # (dim, dim) parity operator

    # --- Precompute X^k, Z^k for k = 0..d-1 ---

    X_pows = [jnp.eye(dim, dtype=dtype)]
    Z_pows = [jnp.eye(dim, dtype=dtype)]
    for k in range(1, dim):
        X_pows.append(X_pows[-1] @ X)
        Z_pows.append(Z_pows[-1] @ Z)
    X_pows = jnp.stack(X_pows, axis=0)  # (d, d, d)
    Z_pows = jnp.stack(Z_pows, axis=0)  # (d, d, d)

    # --- Build A(q,p) = 1/2 Σ_{b1,b2} A^(2d)(2q+b1, 2p+b2) ---
    # A^(2d)(m1,m2) = X^{m1} Z^{m2} P * exp(iπ m1 m2 / d)
    # with exponents modulo d.

    rows = []
    for q_val in range(dim):
        row_ops = []
        for p_val in range(dim):
            A_qp = jnp.zeros((dim, dim), dtype=dtype)

            for b1 in (0, 1):
                for b2 in (0, 1):
                    m1 = 2 * q_val + b1
                    m2 = 2 * p_val + b2

                    k1 = m1 % dim
                    k2 = m2 % dim

                    phase = jnp.exp(1j * jnp.pi * (m1 * m2) / dim)

                    Xk = X_pows[k1]
                    Zk = Z_pows[k2]

                    A_parent = phase * (Xk @ Zk @ P)
                    A_qp = A_qp + A_parent

            A_qp = 0.5 * A_qp
            row_ops.append(A_qp)

        row_ops = jnp.stack(row_ops, axis=0)   # (d, d, d), index p
        rows.append(row_ops)

    A = jnp.stack(rows, axis=0)  # (d, d, d, d), indices (q, p, a, b)
    return A

def phase_point_ops_odd(d: int):
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
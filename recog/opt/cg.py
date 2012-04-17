import numpy as np

def _dense_mult(A, b):
    return np.dot(A, b)

def _sparse_mult(A, b):
    return A*b

def basic_CG(A, b, x0=None, rtol=1e-5, maxiter=200):
    """Solve Ax=b with Conjugate Gradient method, terminate iterations
    if the ratio ||Ax-b||/||b|| < rtol (or if iterations exceed maxiter)

    If A is not a dense ndarray, then it must support the "*" operator
    as a matrix-vector product
    """
    if isinstance(A, np.ndarray):
        prod = _dense_mult
    else:
        prod = _sparse_mult
    if x0 is None:
        x = np.zeros_like(b)
        r = b.copy()
    else:
        x = x0
        r = b - prod(A, x0)
    n = maxiter
    p = r.copy()
    nref = np.dot(b.conj(), b).real
    lrsq = np.dot(r.conj(), r).real
    rtol_sq = rtol**2
    while n > 0:
        delta = lrsq/nref
        if delta < rtol_sq:
            break
        q = prod(A, p)
        alpha = lrsq/np.dot(p.conj(), q).real
        x = x + alpha * p
        r = r - alpha * q
        lrsq_n = np.dot(r.conj(), r).real
        p = r + lrsq_n/lrsq * p
        lrsq = lrsq_n
        n -= 1
    ## if n==0:
    ##     print 'CG failed to converge: final delta=%1.4f'%delta
    ##     return x, maxiter
    return x, maxiter-n


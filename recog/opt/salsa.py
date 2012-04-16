import numpy as np

from cg import basic_CG

def project_to_feasible(s, y, eps):
    sy_err = s - y
    sy_norm = np.dot(sy_err, sy_err)
    if sy_norm > eps**2:
        sy_err *= (eps/np.sqrt(sy_norm))
        sy_err += y
        return sy_err
    return s

# constrained salsa -- solves
# argmin{x} \phi(x) subject to ||Bx - y||_2 <= \epsilon
def c_salsa(B, y, eps, BtB, phi, phi_map, rtol = 1e-5):
    # tolerance of |phi(u_{k}) - phi(u_{k-1])|_1
    # is measured relative to ... backprojected Bt*y ?
    tol = rtol * phi(np.dot(B.T, y))
    # B is (m, n)
    # initialize:
    # d1, v1 size (n,)
    # d2, v2, size (m,)
    m, n = B.shape
    d1 = np.zeros(n)
    v1 = np.zeros(n)
    d2 = np.zeros(m)
    v2 = np.zeros(m)
    u = None

    # create (BtB + I) by adding 1s on the diagonal
    BtB_reg = BtB.copy()
    BtB_reg.flat[::(n+1)] += 1

    J_prev = 1e20
    converged = False
    k = 1
    while not converged:
        r = v1.copy()
        r += d1 + np.dot(B.T, v2+d2)
        # CG solve for u (only a few iterations)
        u = basic_CG(BtB_reg, r, x0=u, maxiter=20)
        Bu = np.dot(B, u)
        v1 = phi_map(u - d1)
        v2 = project_to_feasible(Bu - d2, y, eps)
        d1 += v1 - u
        d2 += v2 - Bu
        # see how much phi(u) has moved
        J = phi(u)
        print 'C-SALSA step', k, 'objective: %1.4f'%J
        if np.abs(J - J_prev) < tol:
            converged = True
        J_prev = J
    return u
        
    

import numpy as np
from scipy.sparse.linalg import LinearOperator
from cg import basic_CG

def project_to_feasible(s, y, eps):
    sy_err = s - y
    sy_norm = np.dot(sy_err, sy_err)
    if sy_norm > eps**2:
        sy_err *= (eps/np.sqrt(sy_norm))
        sy_err += y
        return sy_err
    return s

def indicator(x, y, eps):
    err = x - y
    mse = np.dot(err, err)
    if mse > eps**2:
        return np.inf
    return 0

def regularize_operator(BtB):
    m, m = BtB.shape
    return LinearOperator( (m,m), lambda x: BtB*x + x, dtype='d' )
    
# constrained salsa -- solves
# argmin{x} \phi(x) subject to ||Bx - y||_2 <= \epsilon
def c_salsa(B, Bt, BtBpI_solve, y, eps, phi, phi_map, rtol = 1e-5):
    u = Bt*y
    ## u = np.dot(B.T, y)
    # tolerance of |phi(u_{k}) - phi(u_{k-1])|_1
    # is measured relative to ... backprojected Bt*y ?
    J_prev = phi(u)
    tol = rtol * J_prev

    # B is (m, n)
    # initialize:
    # d1, v1 size (n,)
    # d2, v2, size (m,)
    m, n = B.shape
    d1 = np.zeros(n)
    v1 = u.copy()
    d2 = np.zeros(m)
    v2 = y.copy()


    # create (BtB + I) by adding 1s on the diagonal
    ## BtB_reg = BtB.copy()
    ## BtB_reg.flat[::(n+1)] += 1
    ## BtB_reg = regularize_operator(BtB)

    converged = False
    k = 0
    while not converged:
        k += 1
        ## r = v1 + d1 + np.dot(B.T, v2+d2)
        r = v1 + d1 + Bt*(v2+d2)
        # CG solve for u (only a few iterations)
        u = BtBpI_solve(r, u)
        ## u, _ = basic_CG(BtB_reg, r, x0=u, maxiter=100)
        ## Bu = np.dot(B, u)
        Bu = B*u
        v1 = phi_map(u - d1)
        v2 = project_to_feasible(Bu - d2, y, eps)
        d1 += v1 - u
        d2 += v2 - Bu
        # see how much phi(u) has moved
        J = phi(u) #+ indicator(Bu, y, eps)
        print 'C-SALSA step', k, 'objective: %1.4f'%J, J+indicator(Bu,y,eps)
        ## if np.abs(J - J_prev) < tol:
        ##     converged = True
        ## J_prev = J
        if k == 50:
            converged = True
    ## 0/0
    return u

# second form of SALSA with equality constraint
def c_salsa2(B, Bt, BBt_solve, y, phi, phi_map):

    #u = Bt*y
    Cx = None
    m, n = B.shape
    d = np.zeros(n)
    a = Bt*y #np.zeros(n)
    converged = False
    k = 0
    while not converged:
        k += 1
        u = phi_map(a+d) - d
        Bu = B*u
        ## Cx = BBt_solve(y-Bu, Cx)
        ## d = Bt*Cx
        d = Bt*BBt_solve(y-Bu)
        ## BBt_solve.reset()
        a = d + u
        J = phi(u) #+ indicator(Bu, y, eps)
        print 'C-SALSA2 step',k,'objective: %1.4f'%J,J+indicator(Bu,y,1e-8)
        if k == 50:
            converged = True
    return u
    

import numpy as np
from scipy.sparse.linalg import LinearOperator
from cg import basic_CG
from shrinkers import shrink_thresh

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

def l1_proximity_map(tau, mu):
    tau, mu = map(float, (tau, mu))
    return lambda x: shrink_thresh(x, tau/mu)


def c_salsa_multi(
        B, Bt, BtBpI_solve, ym, eps, phi, phi_map,
        x0=None, n_iter=50, rtol = 1e-5
        ):
    # ym: each column is a RHS vectors, solve for n_col such RHS
    m, n = B.shape
    nrhs = ym.shape[1]
    w = np.empty((n, nrhs), 'd')
    for k in xrange(nrhs):
        x0_k = x0 if x0 is None else x0[:,k]
        w[:,k] = c_salsa(
            B, Bt, BtBpI_solve, ym[:,k], eps, phi, phi_map,
            x0=x0_k, n_iter=n_iter, rtol=rtol
            )
    return w
 
# constrained salsa -- solves
# argmin{x} \phi(x) subject to ||Bx - y||_2 <= \epsilon
def c_salsa(
        B, Bt, BtBpI_solve, y, eps, phi, phi_map,
        x0=None, n_iter=50, rtol = 1e-5, save_sol=False, verbose=False
        ):
    if x0 is None:
        u = Bt*y
    else:
        u = x0

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

    converged = False
    k = 0
    if save_sol:
        du = []
    while not converged:
        k += 1
        ## r = v1 + d1 + np.dot(B.T, v2+d2)
        r = v1 + d1 + Bt*(v2+d2)
        # CG solve for u (only a few iterations)
        ## u = BtBpI_solve(r, u)
        u = BtBpI_solve(r)
        Bu = B*u
        v1 = phi_map(u - d1)
        v2 = project_to_feasible(Bu - d2, y, eps)
        d1 += v1 - u
        d2 += v2 - Bu
        if save_sol:
            du.append(u.copy())
        if k == n_iter:
            converged = True
        # see how much phi(u) has moved
        J = phi(u) #+ indicator(Bu, y, eps)
        r = np.linalg.norm(Bu-y)
        if verbose:
            print 'C-SALSA step', k, 'objective: %1.4f, %1.4e'%(J,r)
        if k>2 and np.abs(J-J_prev) < tol and r < eps:
            converged = True
        J_prev = J
    if save_sol:
        return u, np.array(du)
    return u

def bp_salsa_multi(
        B, Bt, BBt_solve, ym, phi, phi_map,
        x0=None, n_iter=50, rtol=1e-5
        ):
    # ym: each column is a RHS vectors, solve for n_col such RHS
    m, n = B.shape
    nrhs = ym.shape[1]
    w = np.empty((n, nrhs), 'd')
    for k in xrange(nrhs):
        x0_k = x0 if x0 is None else x0[:,k]
        w[:,k] = bp_salsa(
            B, Bt, BBt_solve, ym[:,k], phi, phi_map,
            x0=x0_k, n_iter=n_iter, rtol=rtol #, verbose=True
            )
    return w
# second form of SALSA with equality constraint -- Basis Pursuit SALSA
def bp_salsa(
        B, Bt, BBt_solve, y, phi, phi_map,
        x0=None, n_iter=50, rtol=1e-5, save_sol=False, verbose=False
        ):
    # phi evaluates phi(v)
    # phi_map minimizes phi(v) + mu/2*||d' - v||^2 (e.g. Shrink_{1/mu}(d'))

    ## m, n = B.shape
    ## d = np.zeros(n)
    d = 0
    u = Bt*y if x0 is None else x0

    # tolerance of |phi(u_{k}) - phi(u_{k-1])|_1
    # is measured relative to ... backprojected Bt*y ?
    J_prev = phi(u)
    tol = rtol * J_prev

    converged = False
    k = 0
    if save_sol:
        du = []
    while not converged:
        k += 1
        a = phi_map(u+d) - d
        Ba = B*a
        d = Bt*BBt_solve(y-Ba)
        u = d + a
        if save_sol:
            du.append(u.copy())
        if k == n_iter:
            converged = True
        J = phi(u)
        if verbose:
            r = np.linalg.norm(B*u-y)
            dv = np.linalg.norm(u-a)
            print 'BP-SALSA step',k,'objective: %1.4e, %1.4e, %1.4e'%(J,r,dv)
        if np.abs(J-J_prev) < tol:
            converged = True
        J_prev = J
    if save_sol:
        return u, np.array(du)
    return u

def qreg_salsa_multi(
        B, Bt, BtB_solve, ym, phi, phi_map, mu,
        x0=None, n_iter=50, rtol=1e-5
        ):
    # ym: each column is a RHS vectors, solve for n_col such RHS
    m, n = B.shape
    nrhs = ym.shape[1]
    w = np.empty((n, nrhs), 'd')
    for k in xrange(nrhs):
        x0_k = x0 if x0 is None else x0[:,k]
        w[:,k] = qreg_salsa(
            B, Bt, BtB_solve, ym[:,k], phi, phi_map, mu,
            x0=x0_k, n_iter=n_iter, rtol=rtol
            )
    return w

# Solves quadratic + regularizer form:
# min_x 0.5*||Bx - y||^2 + tau*phi(x)
# which is the Lagrangian of
# min_x phi(x) subject to ||Bx - y||^2 < eps
def qreg_salsa(
        B, Bt, BtB_solve, y, phi, phi_map, mu,
        x0 = None, n_iter = 50, rtol=1e-5, save_sol=False, verbose=False
        ):
    # phi evaluates tau*phi(v)
    # phi_map minimizes the functional tau*phi(v) + mu/2*||d' - v||^2
    # eg, if phi(v) = tau*||v||_1, then phi_map is Shrink_{tau/mu}(d')
    # --> tau is an implicit argument within phi_map
    # --> mu is an implicit argument within BtB_solve and phi_map

    m, n = B.shape
    Bty = Bt*y
    u = 0
    # if x0 is, e.g., the LS solution of ||Bx-y|| then u starts at x0
    v = 0 if x0 is None else x0
    #v = 0
    d = 0
    k = 0

    J_prev = phi(Bty) if x0 is None else phi(x0)
    tol = rtol * J_prev

    if save_sol:
        du = []
    converged = False
    while not converged:
        u = BtB_solve(Bty + mu*(v + d))
        v = phi_map(u-d)
        d = d - u + v
        k = k + 1

        if save_sol:
            du.append(u.copy())
        if k == n_iter:
            converged = True            
        J = phi(u)
        if verbose:
            r = B*u - y
            r = np.linalg.norm(r) #np.dot(r, r)
            dv = np.linalg.norm(u-v) #np.dot(u-v, u-v)
            print 'SALSA step %d l1-norm %1.4f resid %1.2e div %1.2e'%(k, J, r, dv)
        if k>2 and np.abs(J-J_prev) < tol:
            converged = True
        J_prev = J
    if save_sol:
        return u, np.array(du)
    return u

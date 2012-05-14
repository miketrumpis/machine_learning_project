import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sp_la

from ..opt import salsa

def ksvd(
        examples, m, L, mu=2, tau=None, n_iter=20, rtol=1e-4, eps=1e-5
        ):

    n, M = examples.shape
    mu = float(mu)
    
    # initialize A
    A = np.random.randn(n, m)
    c_norm = np.sqrt(np.sum(A**2, axis=0))
    A /= c_norm

    tau = float(tau) if tau else mu / m
    phi = lambda x: np.abs(x).sum()
    phi_map = salsa.l1_proximity_map(tau, mu)

    converged = False
    x0 = None
    k = 1
    while not converged:

        # sparse coding

        # 1) create solver for (AtA + muI) and other ops
        AtA_solve = cholesky_solver(A, mu)
        #opA = lambda x: np.dot(A, x)
        opA = sp_la.LinearOperator(A.shape, A.dot, dtype=A.dtype)
        #opAt = lambda x: np.dot(A.T, x)
        opAt = sp_la.LinearOperator(A.T.shape, A.T.dot, dtype=A.dtype)
        # 2) sparse code each example using Basis Pursuit (Denoising)
        if k > 1:
            x0 = X
        X = salsa.qreg_salsa_multi(
            opA, opAt, AtA_solve, examples, phi, phi_map, mu,
            n_iter = 100, rtol = 1e-5, x0=x0
            )
        if eps == 0:
            top_off_coefs(A, X, examples, L)
        
        # Dictionary update
        for j0 in xrange(m):
            om_j = np.where(np.abs(X[j0]) > eps)[0]
            if not len(om_j):
                continue
            Xt = X[:,om_j]
            Xt[j0] = 0
            Ej0 = examples[:,om_j] - np.dot(A, Xt)
            if Ej0.shape[1] < 2:
                s = la.norm(Ej0[:,0])
                u = Ej0[:,0] / s
                # v is just 1
                vt = np.array([1.0])
            else:
                #u, s, vt = sp_la.svds(Ej0, k=1)
                u, s, vt = la.svd(Ej0, full_matrices=0)
                u = u[:,0]
                s = s[0]
                vt = vt[0]
            A[:,j0] = u.ravel()
            X[j0,om_j] = s*vt.ravel()

        # reject super-redundant atoms,
        # as well as non-representative atoms

        norm_thresh = 0.9
        # maybe it's ok to have the classwse dictionaries
        # have some single-entry atoms
        sparse_thresh = 1
        err = examples - np.dot(A, X)
        err = np.sum(err**2, axis=0)
        AtA = np.dot(A.T, A)
        AtA.flat[::m+1] = 0
        for j in xrange(m):
            Gj = AtA[j]
            Xj = X[j]
            if (Gj.max() > norm_thresh) or \
                (np.sum(np.abs(Xj) > eps) < sparse_thresh):
                best_data = np.argmax(err)
                new_atom = examples[:,best_data]
                a_norm = la.norm(new_atom)
                new_atom /= a_norm
                A[:,j] = new_atom
                X[j,best_data] = a_norm
                err[best_data] = 0
                np.dot(A.T, A, AtA)
                AtA.flat[::m+1] = 0

        cost = err.sum()
        print 'K-SVD step %d, training error %1.4e'%(k,cost)
        if k == 2:
            tol = rtol * cost
            prev_cost = 1e18
        if k > 1 and np.abs(cost - prev_cost) < tol:
            converged = True

        k += 1
        if k > n_iter:
            converged = True
        prev_cost = cost
            
    # end while
    return A, X

        
def top_off_coefs(A, X, Y, L):
    # given a sparsity constraint of L coefs, solve the restricted
    # least squares problem using the columns in A corresponding to
    # the L largest-magnitude coefficients in each column of X
    #
    # NOTE: X changed in-place
    
    n, m = A.shape
    m, M = X.shape
    significance = np.argsort(-np.abs(X), axis=0)
    top_coefs = significance[:L]
    bottom_coefs = significance[L:]
    for mm in xrange(M):
        ym = Y[:,mm]
        cols = top_coefs[:,mm]
        Ar = A[:,cols]
        r = la.lstsq(Ar, ym)
        X[cols,mm] = r[0]
        zcols = bottom_coefs[:,mm]
        X[zcols,mm] = 0
    # done
 
def cholesky_solver(A, mu):
    n, m = A.shape
    AtA = np.dot(A.T, A)
    id = np.eye(m)
    AtA += mu*id
    cho_fac = la.cho_factor(AtA)
    AtA_inv = la.cho_solve(cho_fac, id)
    #return lambda x: np.dot(AtA_inv, x)
    return AtA_inv.dot
    

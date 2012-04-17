import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..opt.cg import basic_CG

# want to form operators dealing with the dictionary frames,
# and the augmented frames for sparse repr and error
#
# The linear operators in this module deal with factorizations of
# the matrix B = [A A_e], where A_e is an m x p transform from a domain
# in which the error is sparse to the image domain. It is required
# that (A_e)(A^T_e) = I_m, with m being the dimensionality of the
# images.

# create operators to perform the following operations for
# B = [A A_e]
# 1) Bw from R^{n+p} --> R^{m}
# 2) (Bt)y from R^{m} --> R^{n+p}
# 3) (BtB)w from R^{n+p} --> R^{n+p}
def Bw(A, e_transform=None):
    m, n = A.shape
    if e_transform is None:
        Ae = lambda x: x
    else:
        Ae = lambda x: e_transform(x)
    # works for matrix-matrix and matrix-vector
    def matvec(w):
        # w in R^{n+p} [x R^{q}] : ie, column(s) are R^{n+p}
        x = w[:n]
        e = w[n:]
        return np.dot(A, x) + Ae(e)
    return matvec

def Bty(At, p, et_transform=None):
    n, m = At.shape
    if et_transform is None:
        Ate = lambda x: x
        p = m
    else:
        Ate = lambda x: et_transform(x)
    # works for matrix-matrix and matrix-vector
    def matvec(y):
        # y in R^{m} [x R^{q}] : ie, column(s) are R^{m}
        if len(y.shape) > 1:
            q = y.shape[1]
            w = np.zeros(n+p, q)
        else:
            w = np.zeros(n+p)
        w[:n] = np.dot(At, y)
        w[n:] = Ate(y)
        ## wx = np.dot(At, y)
        ## we = Ate(y)
        ## return np.vstack( (wx, we) )
        return w
    return matvec

# XXX: this may not be needed
## def BtBw(A, AtA, e_transform=None, et_transform=None):
##     m, n = A.shape
##     if e_transform is None:
##         Ae = lambda x: x
##     else:
##         Ae = lambda x: e_transform(x)
##     if et_transform is None:
##         Ate = lambda x: x
##     else:
##         Ate = lambda x: et_transform(x)
##     def matvec(w):
##         w2 = np.zeros_like(w)
##         x = w[:n]
##         e = w[n:]
##         w2[:n] = np.dot(self.AtA, x)
##         w2[:n] += np.dot(A.T, Ae(e))
##         w2[n:] = e
##         w2[n:] += Ate(np.dot(A, x))
##         return w2
##     return matvec

def diag_loaded_solve(
        A, AtA, e_transforms=(), p=None, **cg_kws
        ):
    # create a method to (approximately) solve (BtB + I)x = y for x
    # The only matrix inverse needed is (I_n + AtA)^-1, however
    # since n grows with the size of the database (training set),
    # then this inverse is assumed to be expensive to compute
    # directly. Therefore, Conjugate Gradients is utilized at this step
    m, n = A.shape
    if not e_transforms or not filter(None, e_transforms):
        E = lambda x: x
        Et = lambda x: x
        p = m
    else:
        E, Et = e_transforms

    AtApI = LinearOperator( (n,n), lambda x: np.dot(AtA, x) + x, dtype='d' )

    # XXX: very numerically unstable!!
    class solver(object):
        c0 = None
        shape = (n+p, n+p)
        def __call__(self, w):
            x = w[:n]
            e = w[n:]

            # precomputed items
            ie = E(e) # p > m
            AtAe_e = np.dot(A.T, ie) # m > n
            AtA_x = np.dot(AtA, x)   # n > n
            

            # this goes to solver
            s = AtA_x + AtAe_e # R^n
            c, iter = basic_CG(AtApI, s, x0=self.c0, **cg_kws)
            self.c0 = c
            print np.linalg.norm(s), iter

            # these are the partitions of the inverse part
            t2 = np.dot(AtA, c)   # n > n
            b2 = Et(np.dot(A, c)) # n > m > p

            # these are the partitions of the BtB_w part
            t1 = s                     # n
            b1 = Et(np.dot(A, x) + ie) # p

            print map(np.linalg.norm, (t1, t2, b1, b2))

            w_out = w.copy()
            w_out[:n] += -t1/2 + t2/4
            w_out[n:] += -b1/2 + b2/4
            return w_out

        def reset(self):
            self.c0 = None

    return solver()
            
            
def BBt_solver(A, AtA, **cg_kws):
    # This variant solves the problem (BBt)x = y for x. Conveniently,
    # the transform domain of the error e is irrelevant, since it
    # is required to be a 1-tight frame (Ae)Ae^t = I_{m}
    # Thus,
    # (BBt)^-1 = I - A[(I_{n} + AtA)^-1]At

    m, n = A.shape
    AtApI = LinearOperator( (n,n), lambda x: np.dot(AtA, x) + x, dtype='d' )

    class solver(object):
        c0 = None
        shape = (m, m)
        def __call__(self, y):
            Aty = np.dot(A.T, y)
            c, it = basic_CG(AtApI, Aty, x0=self.c0, **cg_kws)
            self.c0 = c
            # y_out = I*y - A*x
            y_out = y.copy()
            y_out -= np.dot(A, c)
            return y_out
        def reset(self):
            self.c0 = None
    
    return solver()

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
        return w
    return matvec

def diag_loaded_solve(
        A,
        e_transforms=(),
        mu=1, mxm=True,
        AtA = None,
        **cg_kws
        ):
    from recog import max_inverse_size as max_r
    # create a method to (approximately) solve (BtB + mu*I)x = y for x
    # Depending on whether m<n, then reduce the actual inverse to
    # ((mu+1)I_m + AAt)^-1 (m < n)
    # ((mu+1)I_n + AtA)^-1 (n < m)
    # if the size of the inverse is beyond a certain threshold,
    # use a small number of conjugate gradients iterations to solve
    m, n = A.shape
    r = m if mxm else n
    do_cg = r > max_r
        
    if not e_transforms or not filter(None, e_transforms):
        E = lambda x: x
        Et = lambda x: x
    else:
        E, Et = e_transforms

    if mxm:
        H = np.dot(A, A.T)
    else:
        H = AtA.copy() if AtA is not None else np.dot(A.T, A)
    # make Ci = H + (mu+1)I
    H.flat[::(r+1)] += (mu+1)
    if do_cg:
        Ci = LinearOperator(
            (r,r), lambda x: np.dot(H, x), dtype='d'
            )
        C_solve = lambda x, x0: basic_CG(Ci, x, x0=x0, **cg_kws)[0]
    else:
        C = np.linalg.inv(H)
        C_solve = lambda x, x0: np.dot(C, x)

    mp1_i = 1./(mu+1)
    class mxm_solver(object):
        c0 = None
        ## shape = (n+p, n+p)
        def __call__(self, w):
            x = w[:n]
            e = w[n:]

            # precomputed terms
            ie = E(e)          # p -> m
            Ax = np.dot(A, x)  # n -> m
            u = Ax + ie        # this goes to solver

            c = C_solve(u, self.c0)
            self.c0 = c

            t1 = np.dot(A.T, c)
            b1 = Et(c)

            w_out = w/mu
            w_out[:n] -= t1/mu
            w_out[n:] -= b1/mu
            return w_out
        
    class nxn_solver(object):
        c0 = None
        ## shape = (n+p, n+p)
        def __call__(self, w):
            x = w[:n]
            e = w[n:]

            # precomputed terms
            ie = E(e)          # p -> m
            Ax = np.dot(A, x)  # n -> m
            u = Ax + ie
            s = np.dot(A.T, u) # m -> n
            t = Et(u)          # m -> m (t here not confused with top/bottom)
            
            c = C_solve(s, self.c0)
            self.c0 = c
            Ac = np.dot(A, c)

            # these are the partitions of the inverse part
            t2 = np.dot(A.T, Ac)
            b2 = Et(Ac)

            # these are the partitions of the BtB_w part
            t1 = s           # n
            b1 = Et(Ax + ie) # p

            ## print map(np.linalg.norm, (t1, t2, b1, b2))

            w_out = w.copy()
            w_out[:n] += (t2 - t1)*mp1_i
            w_out[n:] += (b2 - b1)*mp1_i
            w_out /= mu
            return w_out

        def reset(self):
            self.c0 = None

    return mxm_solver() if mxm else nxn_solver()
            
            
def BBt_solver(A, gma=1, mxm = True, AtA = None, **cg_kws):
    from recog import max_inverse_size as max_r
    # This variant solves the problem (BBt)x = y for x. Conveniently,
    # the transform domain of the error e is irrelevant, since it
    # is required to be a 1-tight frame (Ae)Ae^t = I_{m}
    # Thus,
    # (BBt)^-1 = (I_{m} + AAt)^-1              if m < n
    # (BBt)^-1 = I_{m} - A[(I_{n} + AtA)^-1]At if n > m

    m, n = A.shape
    r = m if mxm else n
    do_cg = r > max_r

    if mxm:
        H = np.dot(A, A.T)
    else:
        H = AtA.copy() if AtA is not None else np.dot(A.T, A)

    H.flat[::(r+1)] += gma
    if do_cg:
        C = LinearOperator((r,r), lambda x: np.dot(H, x), dtype = 'd')
        C_solve = lambda x, x0: basic_CG(C, x, x0=x0, **cg_kws)[0]
    else:
        C = np.linalg.inv(H)
        C_solve = lambda x, x0: np.dot(C, x)
    
    class nxn_solver(object):
        c0 = None
        def __call__(self, y):
            Aty = np.dot(A.T, y)
            c = C_solve(Aty, self.c0)
            self.c0 = c
            # y_out = I*y - A*c
            y_out = y.copy()
            y_out -= np.dot(A, c)
            return y_out
        def reset(self):
            self.c0 = None

    class mxm_solver(object):
        c0 = None
        def __call__(self, y):
            #this is just C_solve(y), and done
            yt = C_solve(y, self.c0)
            self.c0 = yt
            return yt
        def reset(self):
            self.c0 = None
    
    return mxm_solver() if mxm else nxn_solver()

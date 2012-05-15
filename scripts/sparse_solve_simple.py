import numpy as np
import matplotlib.pyplot as pp
import recog.dict.facedicts as facedicts
import recog.opt.shrinkers as shrinkers
import recog.opt.salsa as salsa

## trn, tst = facedicts.FacesDictionary.frame_and_samples(
##     'yale_ext', 0.4, 0.6, resize=(10,12)
##     )
trn, tst = facedicts.FacesDictionary.pair_from_saved(
    'downsamp_yale_simple.npz', 0.6, 0.4
    )

m, n = trn.frame.shape
r_col = np.random.randint(0, high=tst.frame.shape[1])
tst_col = tst.frame[:,r_col].ravel()
tst_col -= tst_col.mean()
tst_cls = tst.column_to_class[r_col]
cls_cols = trn.class_to_columns[tst_cls]

# build simple operators A and At for the dictionary
A, At = trn.build_operators()

# find MMSE estimate of regressor weights
r = np.linalg.lstsq(trn.frame, tst_col)
mmse_x = r[0]
if len(r[1]):
    eps = 1.25 * np.sqrt(r[1][0])
else:
    eps = 1.25 * np.linalg.norm( A*mmse_x - tst_col )

# we know that || (muI + AtA)^-1 ||_2 is greater than about 1/mu
# --> || [(muI + AtA)^-1]y ||_2 > (1/mu)*||y||_2 = alpha
# --> so want to perhaps shrink-threshold values < alpha/Ny ??
# so set tau in the range of (k*alpha/n) * mu = k/n
# (for ||y||=1 and A.shape = (m,n))

# want to enforce accuracy pretty heavily
mu = 1.
tau = float(mu) / n
tau = 1/10.
phi1 = lambda x: np.sum(np.abs(x))
phi_map1 = salsa.l1_proximity_map(tau, mu)
AtA_solve = trn.AtApI_solver(mu=mu)

# quadratic + l1 regularize SALSA
x1, u1 = salsa.qreg_salsa(
    A, At, AtA_solve, tst_col, phi1, phi_map1, mu,
    x0=mmse_x, rtol=1e-4, save_sol=True, n_iter=200
    )

# C-SALSA
# let phi_map be the same as above
# make a new solver for (BtB + I)
AtApI_solve = trn.AtApI_solver(mu=1)
mu2 = float(n) / len(cls_cols)
mu2 = 15.
phi_map2 = salsa.l1_proximity_map(1, n)
eps = 5e-3
x2, u2 = salsa.c_salsa(
    A, At, AtApI_solve, tst_col, eps, phi1, phi_map2,
    x0=mmse_x, rtol=1e-4, n_iter=400, save_sol=True
    )

# see how much u is changing
du1 = np.diff(u1, axis=0)
du2 = np.diff(u2, axis=0)
ndu1 = np.array([np.linalg.norm(du) for du in du1])
ndu2 = np.array([np.linalg.norm(du) for du in du2])
del du1, du2
# resids
r1 = A*u1.T - tst_col[:,None]
r1 = np.sqrt((r1**2).sum(axis=0))
r2 = A*u2.T - tst_col[:,None]
r2 = np.sqrt((r2**2).sum(axis=0))
# L1 norm
p1 = np.sum(np.abs(u1), axis=1)
p2 = np.sum(np.abs(u2), axis=1)

pp.figure()
pp.subplot(311)
pp.semilogy(p1)
pp.semilogy(p2)
pp.title('L1 norm')
pp.subplot(312)
pp.semilogy(r1)
pp.semilogy(r2)
pp.axhline(y=eps, color='c', lw=2, ls=':')
pp.title('L2 residual')
pp.subplot(313)
pp.semilogy(p1+r1)
pp.semilogy(p2+r2)
pp.title('Total L1+L2 Cost')

pp.figure()
pp.semilogy(ndu1)
pp.semilogy(ndu2)
pp.title(r"$\|u_{k+1}-u_{k}\|_2$")

pp.figure()
pp.plot(x1)
pp.plot(x2)

pp.show()

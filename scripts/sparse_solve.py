import numpy as np
import matplotlib.pyplot as pp
import recog.dict.facedicts as facedicts
import recog.opt.shrinkers as shrinkers
import recog.opt.salsa as salsa
import recog.faces.distortions as fd

shape = (192,168)
## trn, tst = facedicts.FacesDictionary.frame_and_samples(
##     'yale_ext', 0.6, 0.4
##     )
## trn, tst = facedicts.FacesDictionary.pair_from_saved(
##     'simple_faces', 0.6, 0.4
##     )
## fx = None
## trn, tst = facedicts.RandomFaces.pair_from_saved(
##     'simple_faces', 0.6, 0.4, klass2=facedicts.FacesDictionary,
##     m=200
##     )
## fx = 'rand'
m, n = trn.frame.shape
r_col = np.random.randint(0, high=tst.frame.shape[1])
tst_col = tst.frame[:,r_col].ravel()
tst_col -= tst_col.mean()
fd.random_occlusions(tst_col[:,None], shape, n=3)
if fx=='rand':
    tst_col = np.dot(trn.randomfaces.T, tst_col)
    tst_col /= np.linalg.norm(tst_col)
tst_cls = tst.column_to_class[r_col]
cls_cols = trn.class_to_columns[tst_cls]

# build concatenated operators [A A_e] with identity transform for A_e
B, Bt = trn.build_extended_operators()

# find MMSE estimate of regressor weights
r = np.linalg.lstsq(trn.frame, tst_col)
mmse_x = r[0]
if len(r[1]):
    eps = 1.25 * np.sqrt(r[1][0])
else:
    eps = 1.25 * np.linalg.norm( trn.frame.dot(mmse_x) - tst_col )
x0 = np.zeros( (len(mmse_x) + m,), 'd')
x0[:n] = mmse_x

# we know that || (muI + AtA)^-1 ||_2 is greater than about 1/mu
# --> || [(muI + AtA)^-1]y ||_2 > (1/mu)*||y||_2 = alpha
# --> so want to perhaps shrink-threshold values < alpha/Ny ??
# so set tau in the range of (k*alpha/n) * mu = k/n
# (for ||y||=1 and A.shape = (m,n))

mu = 20. 
tau = float(mu) / n

phi1 = lambda x: np.sum(np.abs(x))
phi_map1 = salsa.l1_proximity_map(tau, mu)
BtB_solve = trn.BtBpI_solver(mu=mu)

# quadratic + l1 regularize SALSA
x1, u1 = salsa.qreg_salsa(
    B, Bt, BtB_solve, tst_col, phi1, phi_map1, mu, n_iter=400,
    x0=x0, verbose=True, save_sol=True, rtol=1e-6
    )

# strict basis pursuit SALSA
# set mu such that the shrinkage threshold is about
# the ratio of class columns to total columns -- i.e., the level
# of a uniform distribution of coefficients.

BBt_solve = trn.BBt_solver()
mu = 100.0
phi_map2 = salsa.l1_proximity_map(1, mu)
x2, u2 = salsa.bp_salsa(
    B, Bt, BBt_solve, tst_col, phi1, phi_map2, n_iter=400, x0=x0,
    verbose=True, save_sol=True, rtol=1e-6
    )
# C-SALSA
# let phi_map be the same as above
# make a new solver for (BtB + I)
eps = 1e-5
BtBpI_solve = trn.BtBpI_solver(mu=1)
phi_map2 = salsa.l1_proximity_map(1, 10.)
x3, u3 = salsa.c_salsa(
    B, Bt, BtBpI_solve, tst_col, eps, phi1, phi_map2, n_iter=400,
    x0=x0, verbose=True, save_sol=True, rtol=1e-6
    )

# see how much u is changing
du1 = np.diff(u1, axis=0)
du2 = np.diff(u2, axis=0)
du3 = np.diff(u3, axis=0)
ndu1 = np.array([np.linalg.norm(du) for du in du1])
ndu2 = np.array([np.linalg.norm(du) for du in du2])
ndu3 = np.array([np.linalg.norm(du) for du in du3])
del du1, du2, du3
# resids
r1 = B*u1.T - tst_col[:,None]
r1 = np.sqrt((r1**2).sum(axis=0))
r2 = B*u2.T - tst_col[:,None]
r2 = np.sqrt((r2**2).sum(axis=0))
r3 = B*u3.T - tst_col[:,None]
r3 = np.sqrt((r3 * r3).sum(axis=0))
# L1 norm
p1 = np.sum(np.abs(u1), axis=1)
p2 = np.sum(np.abs(u2), axis=1)
p3 = np.sum(np.abs(u3), axis=1)

pp.figure()
pp.subplot(311)
pp.semilogy(p1)
pp.semilogy(p2)
pp.semilogy(p3)
pp.title('L1 norm')
pp.subplot(312)
pp.semilogy(r1)
pp.semilogy(r2)
pp.semilogy(r3)
pp.title('L2 residual')
pp.subplot(313)
pp.semilogy(p1+r1)
pp.semilogy(p2+r2)
pp.semilogy(p3+r3)
pp.title('Total L1+L2 Cost')

pp.figure()
pp.semilogy(ndu1)
pp.semilogy(ndu2)
pp.semilogy(ndu3)
pp.title(r"$\|u_{k+1}-u_{k}\|_2$")

pp.figure()
pp.subplot(311)
pp.plot(x1)
pp.subplot(312)
pp.plot(x2)
pp.subplot(313)
pp.plot(x3)

pp.show()

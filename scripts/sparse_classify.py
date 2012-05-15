import random
import numpy as np
import matplotlib.pyplot as pp
import recog.dict.facedicts as facedicts
import recog.opt.shrinkers as shrinkers
import recog.opt.salsa as salsa
import recog.faces.classify as classify

N = 2000
shape = (192,168)
#shape = (12,10)
## trn, tst = facedicts.FacesDictionary.pair_from_saved(
##     'simple_faces', 0.5, 0.5, resize=(16,16)
##     )
## fx = None
## trn, tst = facedicts.EigenFaces.pair_from_saved(
##     'simple_faces', 0.5, 0.5, klass2=facedicts.FacesDictionary,
##     m=120, skip=2
##     )
## fx = 'eig'
trn, tst = facedicts.EigenFaces2.pair_from_saved(
    'simple_faces', 0.5, 0.5, klass2=facedicts.FacesDictionary,
    m=200, skip=0
    )
fx = 'eig'
## trn, tst = facedicts.RandomFaces.pair_from_saved(
##     'simple_faces', 0.4, 0.6, klass2=facedicts.FacesDictionary,
##     m=100
##     )
## fx = 'rand'
## trn, tst = facedicts.DiffusionFaces.pair_from_saved(
##     'downsamp_yale_simple.npz', 0.4, 0.6,
##     klass2=facedicts.FacesDictionary, m=10
##     )

trn.learn_class_dicts(20, 15)

m, n = trn.frame.shape
# choose N test faces to classify
N = min(N, tst.frame.shape[1])
r_cols = np.array(random.sample(xrange(tst.frame.shape[1]), N))

tst_cols = tst.frame[:,r_cols]
tst_cols -= np.mean(tst_cols, axis=0)

# feature transform if necessary
if fx=='eig':
    tst_cols = np.dot(trn.eigenfaces.T, (tst_cols - trn.avg_face[:,None]))
if fx=='rand':
    tst_cols = np.dot(trn.randomfaces.T, tst_cols)
nrm = np.sqrt(np.sum(tst_cols**2, axis=0))
tst_cols /= nrm

tst_classes = [tst.column_to_class[r_col] for r_col in r_cols]

# start with the MMSE reconstructions
r = np.linalg.lstsq(trn.frame, tst_cols)
mmse_x = r[0]
mmse_resids = trn.compute_residuals(mmse_x, tst_cols)
mmse_classes = [min(c)[1] for c in mmse_resids]
mmse_err = [ int( tc != xc )
             for (tc, xc) in zip(tst_classes, mmse_classes) ]
mmse_SCI = trn.SCI(mmse_x)

# allow a little wiggle room for working in the nullspace of A
#eps = 1.25 * np.sqrt(np.max(r[1]))
## eps = 1e-1
## x = classify.classify_faces_dense_err(
##     trn, tst_cols, eps, mu=10.0, x0=mmse_x, n_iter=100, rtol=1e-4
##     )
tau = 10.
x = classify.classify_faces_dense_err_qreg(
    trn, tst_cols, tau, x0=mmse_x, n_iter=100, rtol=1e-5
    )
sparse_resids = trn.compute_residuals(x, tst_cols)
sparse_classes = [min(c)[1] for c in sparse_resids]
sparse_err = [ int( tc != xc )
               for (tc, xc) in zip(tst_classes, sparse_classes) ]
sparse_SCI = trn.SCI(x)

import random
import numpy as np
import matplotlib.pyplot as pp
import recog.dict.facedicts as facedicts
import recog.opt.shrinkers as shrinkers
import recog.opt.salsa as salsa
import recog.faces.classify as classify
import recog.dct2_xforms as dct
import recog.faces.distortions as fd

N = 1200
## N = 50
shape = (192,168)
shape = (96,84)
shape = (27,24)
trn, tst = facedicts.FacesDictionary.frame_and_samples(
    'yale_ext', .6, .4, resize=shape[::-1]
    )
fx = None
## trn, tst = facedicts.FacesDictionary.pair_from_saved(
##     'simple_faces', 0.4, 0.6
##     )
## fx = None
## U, Ut = dct.block_dct_ops(shape, 8)
## trn, tst = facedicts.EigenFaces.pair_from_saved(
##     'simple_faces', 0.6, 0.4, klass2=facedicts.FacesDictionary,
##     m=40, skip=0
##     )
## fx = 'eig'
## trn, tst = facedicts.RandomFaces.pair_from_saved(
##     'simple_faces', 0.6, 0.4, klass2=facedicts.FacesDictionary,
##     m=8000
##     )
## fx = 'rand'
#trn.learn_class_dicts(200, 15)

m, n = trn.frame.shape
# choose N test faces to classify
N = min(N, tst.frame.shape[1])
r_cols = np.array(random.sample(xrange(tst.frame.shape[1]), N))
tst_classes = [tst.column_to_class[r_col] for r_col in r_cols]

tst_cols_clean = tst.frame[:,r_cols]
tst_cols_clean -= np.mean(tst_cols_clean, axis=0)
m1 = tst_cols_clean.shape[0]

num_tests = 9
sig_levels = 1/np.logspace(0, 1, num_tests)
pct_nz_pixels = np.linspace(10,90,num_tests)

mmse_rates = np.zeros(num_tests)
sparse_rates = np.zeros(num_tests)
tests = range(num_tests)
## tests = [0]
for t in tests:
    print 'test', t+1
    # distort test columns in some fashion

    # additive Gaussian noise
    #nz = np.random.randn(m1, N) * np.sqrt(sig_levels[t]/m1)
    #tst_cols = tst_cols_clean.copy() + nz

    # occlusions
    #fd.random_occlusions(tst_cols, shape, n=t+1)

    # pixel corruption
    nc_px = int( np.round(pct_nz_pixels[t]*m1/100) )
    tst_cols = tst_cols_clean.copy()
    for col in tst_cols.transpose():
        px = random.sample(xrange(m1), nc_px)
        ymn = col.min(); ymx = col.max()
        col[px] = (ymx-ymn)*np.random.sample(nc_px) + ymn

    # feature transform if necessary
    if fx=='eig':
        tst_cols = np.dot(trn.eigenfaces.T, (tst_cols - trn.avg_face[:,None]))
    if fx=='rand':
        tst_cols = np.dot(trn.randomfaces.T, tst_cols)

    #nrm = np.sqrt(np.sum(tst_cols**2, axis=0))
    #tst_cols /= nrm
    
    # start with the MMSE reconstructions
    r = np.linalg.lstsq(trn.frame, tst_cols, rcond=1e-8)
    mmse_x = r[0]
    mmse_resids = trn.compute_residuals(mmse_x, tst_cols)
    mmse_classes = [min(c)[1] for c in mmse_resids]
    mmse_err = [ int( tc != xc )
                 for (tc, xc) in zip(tst_classes, mmse_classes) ]
    mmse_err = np.array(mmse_err, dtype='d')
    mmse_SCI = trn.SCI(mmse_x)

    x0 = np.zeros((n+m, N), 'd')
    x0[:n,:] = mmse_x
    for col in xrange(N):
        err = tst_cols[:,col] - np.dot(trn.frame, mmse_x[:,col])
        #x0[n:,col] = U(err)
        x0[n:,col] = err
    # e-transforms are given in
    # (synthesis, analysis) order
    e_xf = (None,None); p = None
    #e_xf = (Ut, U); p = m
    w1 = classify.classify_faces_sparse_err_strict(
        trn, tst_cols, mu=18.0, x0=x0, e_xf=e_xf, p=p, n_iter=100, rtol=1e-6
        )
    x1 = w1[:n,:]
    e1 = w1[n:,:]
    sparse_resids1 = trn.compute_residuals(x1, tst_cols)
    sparse_classes1 = [min(c)[1] for c in sparse_resids1]
    sparse_err1 = [ int( tc != xc )
                    for (tc, xc) in zip(tst_classes, sparse_classes1) ]
    sparse_err1 = np.array(sparse_err1, dtype='d')
    sparse_SCI1 = trn.SCI(x1)

    mmse_rates[t] = 1 - mmse_err.sum()/N
    sparse_rates[t] = 1 - sparse_err1.sum()/N
    


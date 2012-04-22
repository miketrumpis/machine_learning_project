import numpy as np
from ..opt import salsa

def classify_face_sparse_err_strict(frame, atom):
    m, n = frame.frame.shape
    atom = atom / np.linalg.norm(atom)
    B, Bt = frame.build_extended_operators()
    BBt_solve = frame.BBt_solver()
    # get a rough estimate for the number of examples per class
    n_examples = len(frame.class_to_columns.values()[0])
    mu = float(len(atom)) / n_examples
    phi = lambda x: np.sum(np.abs(x))
    phi_map = salsa.l1_proximity_map(1, mu)
    w = salsa.bp_salsa(B, Bt, BBt_solve, atom, phi, phi_map)
    x = w[:n]
    x_err = w[n:]
    resid = frame.compute_residuals(x, atom)
    return sorted(resid)[0], w

def classify_face_sparse_err_loose(frame, atom, mu=10.0, tau=None):
    m, n = frame.frame.shape
    atom = atom / np.linalg.norm(atom)
    B, Bt = frame.build_extended_operators()
    if not tau:
        # since || (BtB+muI)^-1 ||_2 is lower bounded by about mu,
        # and || y ||_2 is normalized to 1,
        # want to shrink values < ||z||_2 / Ny for z = [(BtB+muI)^-1]y
        # set tau to be (k*alpha/Ny) * mu ~ (k*mu**2) / Ny
        tau = 2 * mu**2 / len(atom)
    phi = lambda x: tau*np.sum(np.abs(x))
    phi_map = salsa.l1_proximity_map(tau, mu)
    BtB_solve = frame.BtBpI_solver(mu=mu)
    w = x1 = salsa.qreg_salsa(B, Bt, BtB_solve, atom, phi, phi_map, mu)
    x = w[:n]
    x_err = w[n:]
    resid = frame.compute_residuals(x, atom)
    return sorted(resid)[0], w
    

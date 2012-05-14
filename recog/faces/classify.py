import numpy as np
from ..opt import salsa

def classify_faces_sparse_err_strict(
        frame, atoms, mu=100, x0=None, e_xf=(), p=None,
        **bp_salsa_kw
        ):
    m, n = frame.frame.shape
    B, Bt = frame.build_extended_operators(
        e_transforms=e_xf, p=p
        )
    BBt_solve = frame.BBt_solver()
    # get a rough estimate for the number of examples per class
    #n_examples = len(frame.class_to_columns.values()[0])
    #mu = float(len(atom)) / n_examples
    phi = lambda x: np.sum(np.abs(x))
    phi_map = salsa.l1_proximity_map(1, mu)
    w = salsa.bp_salsa_multi(
        B, Bt, BBt_solve, atoms, phi, phi_map,
        x0=x0, **bp_salsa_kw
        )

    return w
    # XXX: incomplete

def classify_face_sparse_err_loose(frame, atom, mu=10.0, tau=None):
    m, n = frame.frame.shape
    atom = atom / np.linalg.norm(atom)
    B, Bt = frame.build_extended_operators()
    if not tau:
        tau = 2 * float(mu) / n
    phi = lambda x: tau*np.sum(np.abs(x))
    phi_map = salsa.l1_proximity_map(tau, mu)
    BtB_solve = frame.BtBpI_solver(mu=mu)
    w = salsa.qreg_salsa(B, Bt, BtB_solve, atom, phi, phi_map, mu)
    x = w[:n]
    x_err = w[n:]
    resid = frame.compute_residuals(x, atom)
    return sorted(resid)[0], w
    
def classify_faces_dense_err(
        frame, atoms, eps, mu=100.0, x0=None, **c_salsa_kw
        ):
    A, At = frame.build_operators()
    AtApI_solve = frame.AtApI_solver(mu=1)
    m, n = frame.frame.shape
    phi_map = salsa.l1_proximity_map(1, mu)
    phi = lambda x: np.abs(x).sum()
    x = salsa.c_salsa_multi(
        A, At, AtApI_solve, atoms, eps, phi, phi_map,
        x0=x0, **c_salsa_kw
        )
    return x
    # XXX: incomplete

def classify_faces_dense_err_qreg(
        frame, atoms, tau, x0=None, **qreg_salsa_kw
        ):
    A, At = frame.build_operators()
    AtApI_solve = frame.AtApI_solver(mu=1)
    m, n = frame.frame.shape
    mu = 1.0
    phi_map = salsa.l1_proximity_map(tau, mu)
    phi = lambda x: tau*np.abs(x).sum()
    x = salsa.qreg_salsa_multi(
        A, At, AtApI_solve, atoms, phi, phi_map, mu,
        x0=x0, **qreg_salsa_kw
        )
    return x
    # XXX: incomplete


def classify_faces_sparse_err(
        frame, atoms, eps, mu=100, x0=None, e_xf=(), p=None,
        **c_salsa_kw
        ):
    m, n = frame.frame.shape
    B, Bt = frame.build_extended_operators(
        e_transforms=e_xf, p=p
        )
    BtBpI_solve = frame.BtBpI_solver()
    phi = lambda x: np.sum(np.abs(x))
    phi_map = salsa.l1_proximity_map(1, mu)
    w = salsa.c_salsa_multi(
        B, Bt, BtBpI_solve, atoms, eps, phi, phi_map,
        x0=x0, **c_salsa_kw
        )

    return w

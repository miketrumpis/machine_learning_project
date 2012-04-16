import numpy as np

def shrink_thresh(x, alpha, prior=None):
    """Shrinkage-thresholding operator, as defined in [Daub2004], and
    elsewhere.

    This function returns the vector of minimizers of the separable
    objective

    J(v) = l*|v| + (u/2)*||v - x||^2

    with alpha = l/u
    
    Probably changes data in-place
    """
    # XXX: is this shifted proximity operator correct?
    if prior is not None:
        xmask = np.abs(x - prior) > alpha
    else:
        xmask = np.abs(x) > alpha
    if x.dtype in np.sctypes['complex']:
        sx = x / np.abs(x)
    else:
        sx = np.sign(x)
    tx = np.zeros_like(x)
    if isinstance(alpha, np.ndarray):
        tx[xmask] = (np.abs(x[xmask]) - alpha[xmask])*sx[xmask]
    else:
        tx[xmask] = (np.abs(x[xmask]) - alpha)*sx[xmask]
    return tx

def vector_shrinkage(x, alpha):
    "Vector shrinkage operator based on l2 length"
    l2_norm = np.sqrt( np.dot(np.conj(x), x).real )
    scale = 1 - alpha/l2_norm if l2_norm > alpha else 0
    return x*scale

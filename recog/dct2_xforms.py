import numpy as np
import scipy.fftpack as fftpack

from scipy.sparse.linalg import LinearOperator

def block_dct_ops(shape, bsize):
    n = shape[0] * shape[1]
    class bxfm(object):
        ishape = None
        bshape = None
        bsize = None
        @staticmethod
        def fwd_bdct(x):
            x = np.reshape(x, bxfm.ishape)
            xb = A_dct2(
                blockify(x, bxfm.bsize).transpose().copy(), bxfm.bsize
                )
            bxfm.bshape = xb.shape
            return xb.ravel()
        @staticmethod
        def inv_bdct(x):
            x = np.reshape(x, bxfm.bshape)
            return unblock(
                At_dct2(x, bxfm.bsize).transpose(), bxfm.ishape
                ).ravel()
    bxfm.ishape = shape
    bxfm.bsize = bsize
    U = LinearOperator(
        (n,n), bxfm.fwd_bdct, matmat=bxfm.fwd_bdct, dtype='d'
        )
    Ut = LinearOperator(
        (n,n), bxfm.inv_bdct, matmat=bxfm.inv_bdct, dtype='d'
        )
    return bxfm.fwd_bdct, bxfm.inv_bdct

def A_dct2(x, n, omega=None):
    """
    Take the 2-dimensional type II DCT of the flattened (n,n) image
    contained in vector x. Works across columns.

    Parameters
    ----------

    x : ndarray, shape (n*n,n_col)
      flattened image vector
    n : int
      image column/row size
    omega : ndarray
      support of the output (ie, indices at which output vector is sampled)

    Returns
    -------

    y = dct2(x.reshape(n,n))[omega]
    """
    col_shapes = x.shape[1:] if len(x.shape) > 1 else ()
    x.shape = (n,n) + col_shapes
    y = fftpack.dct(x, type=2, axis=0, norm='ortho')
    y = fftpack.dct(y, type=2, axis=1, norm='ortho')
    x.shape = (n*n,) + col_shapes
    # this syntax needed because y is discontiguous in memory
    y = np.reshape(y, x.shape)
    if omega:
        return np.take(y, omega, axis=0)
    return y

def At_dct2(y, n, omega=None):
    """
    Take the 2-dimensional type III DCT of the flattened (n,n) matrix
    contained in vector y. This is the adjoint operator to the A_dct2
    operator defined above. Omega is the support over the n**2
    dimensional space of the flattened DCT coefficients.
    """
    col_shapes = y.shape[1:] if len(y.shape) > 1 else ()
    if omega:
        y2 = np.zeros((n*n,)+col_shapes, 'd')
        y2[omega] = y
    else:
        y2 = y
    y2.shape = (n,n) + col_shapes
    w = fftpack.dct(y2, type=3, axis=0, norm='ortho')
    w = fftpack.dct(w, type=3, axis=1, norm='ortho')
    y2.shape = (n*n,) + col_shapes
    return w.reshape( (n*n,) + col_shapes )
    
def blockify(image, bsize, mode='pad'):
    # convert to a sequence of bsize*bsize vectors,
    # zero pad or truncate if necessary
    if image.ndim > 2:
        raise ValueError('image must be grayscale: ndim == 2')
    Ny, Nx = image.shape
    if mode.lower()=='pad':
        nh_blocks = int(np.ceil(Nx/bsize))
        nv_blocks = int(np.ceil(Ny/bsize))
        if nh_blocks*bsize > Nx or nv_blocks*bsize > Ny:
            pimage = np.zeros((nv_blocks*bsize, nh_blocks*bsize), image.dtype)
            pimage[:Ny,:Nx] = image
            image = pimage
    else:
        nh_blocks = int(Nx/bsize)
        nv_blocks = int(Ny/bsize)
        image = image[:nv_blocks*bsize, :nh_blocks*bsize]
    # the strides within a block are equal to the array's strides
    sy, sx = image.strides
    # the strides indexing different blocks are bsize times the original strides
    isy, isx = map(lambda x: bsize*x, (sy, sx))
    blk_image = np.lib.stride_tricks.as_strided(
        image, shape=(nv_blocks, nh_blocks, bsize, bsize),
        strides=(isy, isx, sy, sx)
        )
    return np.reshape( blk_image, (nh_blocks*nv_blocks, bsize**2) )

def unblock(blk_image, shape):
    bsize = np.sqrt(blk_image.shape[1])
    Ny, Nx = shape
    nh_blocks = int(np.ceil(Nx/bsize))
    nv_blocks = int(np.ceil(Ny/bsize))
    image = np.empty((nv_blocks*bsize, nh_blocks*bsize), blk_image.dtype)
    # the strides within a block are equal to the array's strides
    sy, sx = image.strides
    # the strides indexing different blocks are bsize times the original strides
    isy, isx = map(lambda x: bsize*x, (sy, sx))
    blk_view = np.lib.stride_tricks.as_strided(
        image, shape=(nv_blocks, nh_blocks, bsize, bsize),
        strides=(isy, isx, sy, sx)
        )
    blk_view[:,:,:,:] = np.reshape(
        blk_image, (nv_blocks, nh_blocks, bsize, bsize)
        )
    if nh_blocks*bsize > Nx or nv_blocks*bsize > Ny:
        return image[:Ny, :Nx].copy()
    return image

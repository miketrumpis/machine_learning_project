import numpy as np

def random_occlusions(ym, shape, n=1):
    # apply n random distortions to each image-column in y
    # in the form of rectangles with random width/height

    n_col = ym.shape[1]
    h, w = shape

    rects = np.abs(np.random.multivariate_normal([75,70], np.array([[40, .15],[.15, 40]]), (n_col, n)))
    rects = np.round(rects).astype('i')
    
    x0 = np.floor(np.random.uniform(high=w, size=(n_col,n))).astype('i')
    y0 = np.floor(np.random.uniform(high=h, size=(n_col,n))).astype('i')

    for k in xrange(n_col):
        y = ym[:,k]
        y.shape = shape
        for nn in xrange(n):
            xx = x0[k,nn]
            yy = y0[k,nn]
            rx = min(w-1, max(0, rects[k,nn,0]))
            ry = min(h-1, max(0, rects[k,nn,1]))
            y[yy:yy+ry, xx:xx+rx] = 0
    
    

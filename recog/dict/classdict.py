import numpy as np
from scipy.sparse.linalg import LinearOperator
from recog.support.descriptors import auto_attr, ResetMixin
import block_factorizing as bf
import ksvd
from recog import scratch
import os

class ClassDictionary(ResetMixin):
    """
    This dictionary will be the combination of linear frame basis, and
    a correspondence between the columns and the L classes composing
    the dictionary elements.
    """

    def __init__(
            self, database, cls_order=None, dtype='d', debias=True,
            **other_kws
            ):
        """
        Construct a dictionary (frame) from a database (Python dict,
        or preconstructed matrix).
        Keep track of which columns correspond to which classes
        """

        if isinstance(database, dict):
            self._init_from_dict(database, dtype, debias)
        else:
            self._init_from_array(database, cls_order, dtype, debias)
        self.n_classes = len(self.class_to_columns)
        self._normalize_frame()

    def _normalize_frame(self):
        sq_norms = np.sum(self.frame**2, axis=0)
        self.frame /= np.sqrt(sq_norms)

    def _init_from_dict(self, database, dtype, debias):
        # structure of database is a list of ndarrays per key ...
        # flatten the arrays and concatenate them columnwise to
        # form the frame. Keep track of class-to-columns and
        # column-to-class correspondences
        n = 0
        class_to_columns = dict()
        column_to_class = dict()
        frame = list() # going to stack columns row-wise first
        for cls, arr in database.iteritems():
            # arr is a list of n examples
            # array is (n_examples, edim1, [edim2, ...])
            n_cols = len(arr)
            cols = np.array(arr, dtype=dtype).reshape(n_cols, -1)
            if debias:
                c_means = cols.mean(axis=1)
                cols -= c_means[:,None]
            
            frame.append(cols)
            # XXX: this tuple is causing problems in indexing.. why??
            #col_nums = tuple(range(n, n+n_cols))
            col_nums = range(n, n+n_cols)
            class_to_columns[cls] = col_nums
            column_to_class.update( ((c, cls) for c in col_nums) )
            n += n_cols
        frame = np.vstack(frame)
        frame = frame.transpose()
        del database
        self.frame = frame
        self.class_to_columns = class_to_columns
        self.column_to_class = column_to_class

    def _init_from_array(self, database, cls_order, dtype, debias):
        # cls_order is simply class_to_columns
        if database.dtype != dtype:
            database = database.astype(dtype)
        if debias:
            c_means = database.mean(axis=0)
            database -= c_means
        self.frame = database
        self.class_to_columns = cls_order
        self.column_to_class = dict()
        for cls, cols in self.class_to_columns.iteritems():
            self.column_to_class.update( ((c, cls) for c in cols) )

    def learn_class_dicts(self, m, L):
        # learn a separate dictionary for each class of samples
        # in the current dictionary
        new_frame = []
        for cls, cols in self.class_to_columns.iteritems():
            print 'learning class', cls, 'from %d examples'%len(cols)
            Y = self.frame[:,cols]
            A, X = ksvd.ksvd(Y, m, L, n_iter=60)
            n_skip = m*len(new_frame)
            self.class_to_columns[cls] = range(n_skip, n_skip + m)
            new_frame.append(A)
        new_col_to_class = dict()
        for cls, cols in self.class_to_columns.iteritems():
            new_col_to_class.update( ((c, cls) for c in cols) )
        self.column_to_class = new_col_to_class
        self.frame = np.hstack(new_frame)
        self._normalize_frame()

    @classmethod
    def pair_from_saved(
            klass, fname, training, testing,
            klass2=None, **klass_kws
            ):
        """
        Return a training/testing dictionary from a saved matrix.

        Parameters
        ----------

        fname : str
          the filename, to be found in the scratch directory

        training : float
          proportion of columns to use for training

        testing : float
          proportion of columns to use for testing (training + testing <= 1)

        Notes
        -----

        doc test
        >>> r = classdict.ClassDictionary.pair_from_saved('asdf2', .4, .2)
        >>> set(r[0]).isdisjoint(set(r[1]))
        True
        >>> sum([ len(v) for v in r[3].values() ]) == len(r[1])
        True
        >>> sum([ len(v) for v in r[2].values() ]) == len(r[0])
        True
        """

        fname = os.path.splitext(fname)[0] + '.npz'
        arrz = np.load(os.path.join(scratch, fname))
        # this is a run-length descriptions of the columns per class
        cls_counts = arrz['arr_0']
        # expand it into column indices
        trn_cls_columns = list()
        trn_cols_by_class = dict()
        tst_cls_columns = list()
        tst_cols_by_class = dict()
        # XXX: The following logic seems redundant with (and perhaps
        # faster than) the partitioning in "load_faces()". Think about
        # making a "partition matrix" function
        n = 0
        n_trn = 0
        n_tst = 0
        for item in cls_counts:
            cls = item[0]; count = item[1]
            c_trn = int( np.ceil( training*count ) )
            c_tst = int( np.floor( testing*count ) )
            shuff_idx = np.arange(n, n+count)
            n += count
            np.random.shuffle(shuff_idx)
            
            trn_cls_columns = trn_cls_columns + \
                list(shuff_idx[:c_trn])
            tst_cls_columns = tst_cls_columns + \
                list(shuff_idx[c_trn:c_trn+c_tst])

            trn_cols_by_class[cls] = range(n_trn, n_trn + c_trn)
            n_trn += c_trn
            tst_cols_by_class[cls] = range(n_tst, n_tst + c_tst)
            n_tst += c_tst

        print 'built partitions'
        # now partition the matrix into training and testing sets
        matrix = arrz['arr_1']
        print 'loaded matrix'
        trn_matrix = matrix[:, trn_cls_columns]
        tst_matrix = matrix[:, tst_cls_columns]
        print 'partitioned matrix'
        training = klass(
            trn_matrix, cls_order=trn_cols_by_class,
            dtype=trn_matrix.dtype, **klass_kws
            )
        if not klass2:
            klass2 = klass
        testing = klass2(
            tst_matrix, cls_order=tst_cols_by_class,
            dtype=tst_matrix.dtype, **klass_kws
            )
        print 'built classes'
        return training, testing
    
        ## return trn_cls_columns, tst_cls_columns, trn_cols_by_class, tst_cols_by_class

    @auto_attr
    def AtA(self):
        return np.dot(self.frame.T, self.frame)

    def __repr__(self):
        n, m = self.frame.shape
        classes = len(self.class_to_columns)
        return 'A %d x %d frame of %d classes'%(n, m, classes)

    def build_operators(self):
        """
        Construct LinearOperators for the matrix products Ax and Aty
        """
        A = self.frame
        m, n = A.shape
        Ax = lambda x: np.dot(A, x)
        Ax = LinearOperator( (m,n), Ax, matmat=Ax, dtype='d' )
        Aty = lambda y: np.dot(A.T, y)
        Aty = LinearOperator( (n,m), Aty, matmat=Aty, dtype='d' )
        return Ax, Aty

    def AtApI_solver(self, mu=1, **cg_kws):
        """
        Construct the solver for (AtA + muI)x = y (for x).

        The inverse is carried out on a matrix of rank min(m,n),
        where m x n is the size of A
        """
        from recog import max_inverse_size as max_r
        A = self.frame
        m, n = A.shape
        r = min(m,n)
        do_cg = r > max_r

        if m < n:
            H = np.dot(A, A.T)
        else:
            H = self.AtA.copy()
        H.flat[::(r+1)] += mu
        if do_cg:
            C = LinearOperator((r,r), lambda x: np.dot(H, x), dtype = 'd')
            C_solve = lambda x, x0: basic_CG(C, x, x0=x0, **cg_kws)[0]
        else:
            C = np.linalg.inv(H)
            C_solve = lambda x, x0: np.dot(C, x)

        class mxm_solve(object):
            c0 = None
            def __call__(self, x):
                s = np.dot(A, x)
                c = C_solve(s, self.c0)
                s = np.dot(A.T, c)
                x_out = (x - s)
                x_out /= mu
                return x_out
        class nxn_solve(object):
            c0 = None
            def __call__(self, x):
                self.c0 = C_solve(x, self.c0)
                return self.c0
        
        return nxn_solve() if n < m else mxm_solve()

    def build_extended_operators(
            self, e_transforms = (None, None), p = None
            ):
        """
        Construct the LinearOperators which correspond to B = [A A_e]
        in the problem y = Ax + e' = [A A_e]*[x; e] = Bw

        Specifically, these operators compute B*w, (Bt)*y
        (and BtB*w ???)
        """
        
        m, n = self.frame.shape
        e_transform, et_transform = e_transforms
        p = m if not p else p
        Bw = bf.Bw(self.frame, e_transform)
        Bty = bf.Bty(self.frame.T, p, et_transform)
        B = LinearOperator(
            (m, n+p), Bw, matmat=Bw, dtype='d'
            )
        Bt = LinearOperator(
            (n+p, m), Bty, matmat=Bty, dtype='d'
            )
        return B, Bt

    def BtBpI_solver(self, **kwargs):
        """
        Constructs an operator to solve (mu*I + BtB)x = y for x
        """

        # (muI + BtB)^-1 = umI - Bt*[(umI + BBt)^-1]*B
        #                = umI - Bt*[((um+1)I + AAt)^-1]*B ((A_e)(A_et) = I)
        # where um = 1/mu
        # two paths -- if m < n, then solve this problem
        # otherwise, if m > n, decompose the inner inverse again
        A = self.frame
        AtA = self.AtA
        m, n = A.shape
        mxm = m < n
        #mxm = False
        return bf.diag_loaded_solve(A, AtA=AtA, mxm = mxm, **kwargs)

    def BBt_solver(self, **kwargs):
        """
        Constructs an operator to solve (BBt)u = w for u
        """
        A = self.frame
        AtA = self.AtA
        m, n = A.shape
        return bf.BBt_solver(A, mxm = m < n, AtA = AtA, **kwargs)

    def compute_residuals(self, x, y):
        """
        Return a sequence of pairs (r_i, cls_i) where r_i is the
        l2 residual ||AR(i)x - y||^2, and R(i) is a projection to
        the subspace of class i
        """
        # return one list per column in x,y
        resids = []
        if len(x.shape) < 2:
            x = np.reshape(x, (len(x), -1))
            y = np.reshape(y, (len(y), -1))
        for rhs in xrange(x.shape[1]):
            resids.append( list() )
        for cls, cols in self.class_to_columns.iteritems():
            #xi = np.take(x, cols)
            xi = x[cols]
            Ai = self.frame[:, cols]
            ri = np.dot(Ai,xi) - y
            for rhs in xrange(ri.shape[1]):
                resids[rhs].append( (np.dot(ri[:,rhs], ri[:,rhs]), cls) )
        if len(resids) < 2:
            return resids[0]
        return resids

    def SCI(self, x):
        k = self.n_classes
        ## ell1_i = [ np.abs(np.take(x, cols)).sum()
        ##            for cols in self.class_to_columns.itervalues() ]
        ell1_i = [ np.abs(x[cols]).sum(axis=0)
                   for cols in self.class_to_columns.itervalues() ]
        mx_di = np.max(ell1_i, axis=0)
        ell1 = np.sum(ell1_i, axis=0)
        return (k*mx_di/ell1 - 1) / (k-1)
        

def save_whole_dictionary(partitions, fname):
    # the partitions sequence should hold a number of ClassDictionaries,
    # whose class sets are identical, but whose columns represent distinct
    # samples from those classes.
    # Note -- No check is performed to enforce the uniqueness of the
    # samples!! Neither are the consistencies of the samples checked (i.e.
    # whether the bias and norm are standardized)

    # build a new matrix whose contiguous column partitions are a complete
    # set of samples from each class. Columns within classes in arbitrary
    # order, and then classes in arbitrary order

    p1 = partitions[0]
    m = p1.frame.shape[0]
    cls_set = set( p1.class_to_columns.keys() )
    cls_cols = dict(( (c, []) for c in cls_set ))
    for p in partitions:
        c_set = set( p.class_to_columns.keys() )
        # should make sure intersection is complete
        if cls_set.difference(c_set):
            del cls_cols
            raise ValueError('One of the partitions includes a new class')
        for cls in c_set:
            c_cols = p.class_to_columns[cls]
            cls_cols[cls].append( p.frame[:, c_cols] )
    # flatten the list of column partitions, then join
    n_cols = [ c[0].shape[1] + c[1].shape[1] for c in cls_cols.itervalues() ]
    mat = np.empty( (m, np.sum(n_cols)), p1.frame.dtype )
    n = 0
    cls_order = []
    for cls, cols in cls_cols.iteritems():
        n1 = cols[0].shape[1]
        n2 = cols[1].shape[1]
        mat[:,n:n+n1] = cols[0]; n += n1
        mat[:,n:n+n2] = cols[1]; n += n2
        cls_order.append( (cls, n1+n2) )

    descr_arr = np.asanyarray(cls_order, dtype=object)
    ## dtype = np.dtype( [ ('class_counts', object), ('matrix', object) ] )
    ## aug_mat = np.empty(1, dtype=dtype)
    ## aug_mat['class_counts'][0] = np.asarray(cls_order)
    ## aug_mat['matrix'][0] = mat

    ## return mat, cls_order
    fname = os.path.join(scratch, fname)
    fname = os.path.splitext(fname)[0]
    np.savez(fname, descr_arr, mat)
    return fname

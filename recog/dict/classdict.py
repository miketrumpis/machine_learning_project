import numpy as np
from scipy.sparse.linalg import LinearOperator
from recog.support.descriptors import auto_attr
## from .block_factorizing import Bw, Bty, diag_loaded_solve
import block_factorizing as bf
from recog import scratch
import os

class ClassDictionary(object):
    """
    This dictionary will be the combination of linear frame basis, and
    a correspondence between the columns and the L classes composing
    the dictionary elements.
    """

    def __init__(self, database, cls_order=None, dtype='d', debias=True):
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
            col_nums = tuple(range(n, n+n_cols))
            class_to_columns[cls] = col_nums
            column_to_class.update( ((c, cls) for c in col_nums) )
            n += n_cols
        frame = np.vstack(frame)
        frame = frame.transpose()
        del database
        sq_norms = np.sum(frame**2, axis=0)
        self.frame = frame / np.sqrt(sq_norms)
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
        for cls, cols in cls_order.iteritems():
            self.column_to_class.update( ((c, cls) for c in cols) )

    @classmethod
    def pair_from_saved(klass, fname, training, testing):
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
            trn_matrix, cls_order=trn_cols_by_class, dtype=trn_matrix.dtype
            )
        testing = klass(
            tst_matrix, cls_order=tst_cols_by_class, dtype=tst_matrix.dtype
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
        B = LinearOperator(
            (m, n+p), bf.Bw(self.frame, e_transform), dtype='d'
            )
        Bt = LinearOperator(
            (n+p, m), bf.Bty(self.frame.T, p, et_transform), dtype='d'
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
        return bf.diag_loaded_solve(A, AtA=AtA, mxm = m < n, **kwargs)

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
        resids = []
        for cls, cols in self.class_to_columns.iteritems():
            xi = np.take(x, cols)
            Ai = np.take(self.frame, cols, axis=1)
            ri = np.dot(Ai,xi) - y
            resids.append( (np.dot(ri, ri), cls) )
        return resids

    def SCI(self, x, cls):
        k = self.n_classes
        mx_di = np.take(x, self.class_to_columns[cls]).max()
        ell1 = np.sum(np.abs(x))
        return (k*mx_di/ell1 - 1) / (k-1)
        

def save_whole_dictionary(partitions, fname):
    # the partitions sequence should hold a number of ClassDictionaries,
    # whose class sets are identical, but whose columns represent distinct
    # samples from those classes.
    # Note -- No check is performed to enforce the uniqueness of the
    # samples!! Niether are the consistencies of the samples checked (i.e.
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

import numpy as np
from scipy.sparse.linalg import LinearOperator
from recog.image import load_faces
from recog.support.descriptors import auto_attr
## from .block_factorizing import Bw, Bty, diag_loaded_solve
import block_factorizing as bf
class ClassDictionary(object):
    """
    This dictionary will be the combination of linear frame basis, and
    a correspondence between the columns and the L classes composing
    the dictionary elements.
    """

    def __init__(self, database, dtype='d', debias=True):
        """
        Construct a dictionary (frame) from a database (Python dict).
        Keep track of which columns correspond to which classes
        """

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

    @auto_attr
    def AtA(self):
        return np.dot(self.frame.T, self.frame)

    def __repr__(self):
        n, m = self.frame.shape
        classes = len(self.class_to_columns)
        return 'A %d x %d frame of %d classes'%(n, m, classes)

    def build_operators(
            self, e_transforms = (None, None), p = None, **cg_kws
            ):        
        m, n = self.frame.shape
        e_transform, et_transform = e_transforms
        p = m if not p else p
        B = LinearOperator(
            (m, n+p), bf.Bw(self.frame, e_transform), dtype='d'
            )
        Bt = LinearOperator(
            (n+p, m), bf.Bty(self.frame.T, p, et_transform), dtype='d'
            )
        ## BtB = LinearOperator(
        ##     (n+p, n+p), self.BtBw(), dtype='d'
        ##     )
        # this is very unstable!!
        ## BtBpI_solver = bf.diag_loaded_solve(
        ##     self.frame, self.AtA, e_transforms=e_transforms, p=p, **cg_kws
        ##     )
        ## from ..opt.cg import basic_CG
        ## # this is very slow :(
        ## BtBpI = LinearOperator(
        ##     (n+p, n+p), lambda x: x + Bt*(B*x), dtype='d'
        ##     )
        ## BtBpI_solver = lambda x, x0: basic_CG(BtBpI, x, x0=x0, **cg_kws)[0]
        BBt_solve = bf.BBt_solver(self.frame, self.AtA, **cg_kws)
        return B, Bt, BBt_solve


# XXX: this may become PlainFacesDictionary later
class FacesDictionary(ClassDictionary):

    @staticmethod
    def frame_and_samples(dbname, training, testing, **class_kws):
        training, testing = load_faces(dbname, training, testing)
        trn_dict = FacesDictionary(training, **class_kws)
        tst_dict = FacesDictionary(testing, debias=False)
        return trn_dict, tst_dict
    

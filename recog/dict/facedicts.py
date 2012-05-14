from classdict import ClassDictionary
from ..image import load_faces

import numpy as np

# XXX: this may become PlainFacesDictionary later
class FacesDictionary(ClassDictionary):

    @staticmethod
    def frame_and_samples(
            dbname, training, testing, resize=(), **class_kws
            ):
        training, testing = load_faces(
            dbname, training, testing, resize=resize
            )
        trn_dict = FacesDictionary(training, **class_kws)
        tst_dict = FacesDictionary(testing, **class_kws)
        return trn_dict, tst_dict
    
class EigenFaces(FacesDictionary):
    """
    Projects all training set faces onto the first [skip,skip+m)
    'eigenfaces'. Keep these projection coefficients as new
    m-dimensional features.
    """
    def __init__(self, *args, **kwargs):
        m = kwargs.pop('m', 10)
        skip = kwargs.pop('skip', 2)
        FacesDictionary.__init__(self, *args, **kwargs)
        self.avg_face = np.mean(self.frame, axis=1)
        self.frame -= self.avg_face[:,None]
        print 'svd'
        [_,_,Vt] = np.linalg.svd(self.frame, full_matrices=0)
        pre_xform = Vt[skip:skip+m].transpose()
        print 'feature xform'
        eigenfaces = np.dot(self.frame, pre_xform)
        # for each m1 x p_i collection of training set faces for
        # the i^th class, transform to m x p_i
        eigenframe = np.dot(eigenfaces.T, self.frame)
        self.frame = eigenframe
        self._normalize_frame()
        self.eigenfaces = eigenfaces

class RandomFaces(FacesDictionary):
    """
    Projects all training set faces onto m Gaussian random vectors.
    """
    def __init__(self, *args, **kwargs):
        m2 = kwargs.pop('m', 10)
        FacesDictionary.__init__(self, *args, **kwargs)
        m, n = self.frame.shape

        self.randomfaces = np.random.randn(m, m2)
        print 'feature xform'
        self.frame = np.dot(self.randomfaces.T, self.frame)
        self._normalize_frame()

        

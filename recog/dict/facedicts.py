from classdict import ClassDictionary
from ..image import load_faces

# XXX: this may become PlainFacesDictionary later
class FacesDictionary(ClassDictionary):

    @staticmethod
    def frame_and_samples(dbname, training, testing, **class_kws):
        training, testing = load_faces(dbname, training, testing)
        trn_dict = FacesDictionary(training, **class_kws)
        tst_dict = FacesDictionary(testing, debias=False)
        return trn_dict, tst_dict
    
class EigenFaces(FacesDictionary):
    pass

class RandomFaces(FacesDictionary):
    pass

# this might be Eigenfaces
class PCAFaces(FacesDictionary):
    pass

class DiffusionFaces(FacesDictionary):
    pass

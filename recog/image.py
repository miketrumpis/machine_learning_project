import numpy as np
import random
import os.path as p
from glob import glob
from PIL import Image

def load_faces(dataset, training_set, testing_set, resize=()):
    # if training/testing are floats < 1, then they are proportions
    # of the total data to be chosen randomly
    # resize is given as (w, h)
    if isinstance(training_set, float) and isinstance(testing_set, float):
        # the sum of the fractions should be <= 1
        assert (1 - training_set - testing_set) > -1e-8, 'Bad proportions'
        training_frac = training_set
        testing_frac = testing_set
    else:
        raise NotImplementedError(
            'Only supporting random fractional partitions'
            )
    # get dictionaries of images keyed by class labels
    if dataset == 'yale_ext':
        images_by_class = load_yale(resize=resize)
    elif dataset == 'ar_faces':
        images_by_class = load_ar_faces()
    else:
        raise ValueError('dataset %s not recognized'%dataset)

    training_images = dict()
    testing_images = dict()
    for cls, images in images_by_class.iteritems():
        n = len(images)
        n_trn = int( np.ceil( training_frac*n ) )
        n_tst = int( np.floor( testing_frac*n ) )
        r_idx = random.sample(xrange(n), n)
        training_images[cls] = [images[i] for i in r_idx[:n_trn]]
        testing_images[cls] = [images[i] for i in r_idx[n_trn:n_trn+n_tst]]
    del images_by_class
    return training_images, testing_images

def load_saved_faces(dname):
    pass

def load_yale(
        exclude_ambient=True, raise_if_inhomogeneous=True,
        resize=()
        ):
    # I guess importing config parameters locally is one way of
    # making them mutable... but dumb
    from recog import yale_ext
    
    class_dirs = glob(p.join(yale_ext, 'yaleB*'))
    images = dict()
    for cls in class_dirs:
        pgm_files = glob(p.join(cls, '*.pgm'))
        if exclude_ambient:
            pgm_files = filter(lambda s: s.find('Ambient') < 0, pgm_files)
        cls_label = p.split(cls)[1]
        # just keep the string of the last 2 digits
        cls_label = cls_label[-2:]
        if resize:
            examples = [np.array(Image.open(pgm).resize(resize))
                        for pgm in pgm_files]
        else:
            examples = [np.array(Image.open(pgm)) for pgm in pgm_files]
        if raise_if_inhomogeneous and \
            filter( lambda a: a.shape != examples[0].shape, examples[1:] ):
            raise ValueError(
                'The atoms for class %s are not consistently sized'%cls_label
                )
        images[cls_label] = examples
    return images

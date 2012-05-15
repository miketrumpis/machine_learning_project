import random
import matplotlib.pyplot as pp
import recog.dict.facedicts as facedicts

trn, tst = facedicts.FacesDictionary.pair_from_saved(
    'simple_faces', 0.1, 0.1
    )
shape = (192,168)

rn = random.sample(xrange(trn.frame.shape[1]), 4)
for n in xrange(4):
    f = pp.figure()
    pp.imshow(trn.frame[:,rn[n]].reshape(shape), cmap=pp.cm.gray)
    pp.gca().xaxis.set_visible(False)
    pp.gca().yaxis.set_visible(False)
    pp.draw()
    f.savefig('dictfigs/face_%d.pdf'%(n+1,))

    f = pp.figure()
    pp.imshow(
        trn.frame[::3000,rn[n]].reshape(11,1),
        cmap=pp.cm.gray, interpolation='nearest'
        )
    pp.gca().xaxis.set_visible(False)
    pp.gca().yaxis.set_visible(False)
    f.savefig('dictfigs/col_%d.pdf'%(n+1,))

pp.show()

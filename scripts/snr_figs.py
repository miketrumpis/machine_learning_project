import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pp

files = (
    'snr_trial_40f.npz', 'snr_trial_80f.npz',
    'snr_trial_160f.npz', 'snr_trial_320f.npz'
    )
num_tests = 8

mmse_err = np.zeros((len(files),num_tests))
sparse_err = np.zeros((len(files),num_tests))

sig_levels = 1/np.logspace(0, 1, num_tests)

for n, f in enumerate(files):
    arr = np.load(f)
    mmse_err[n] = arr['mmse_rates']
    sparse_err[n] = arr['sparse_rates']

snr_db = 10*np.log10(1/sig_levels)
fdim = np.array([40, 80, 160, 320], 'd')

xpos, ypos = np.meshgrid(fdim, snr_db)

dy = np.c_[ypos[1].T*.75, np.diff(ypos, axis=0).T*.75].T
dy = dy
#dx = np.c_[xpos[:,0]*.75, np.diff(xpos, axis=1)*.75]
dx = np.ones_like(dy)
dx.fill(18.)

azim = -112
elev = 9.4

f = pp.figure()
ax = f.add_subplot(111, projection='3d')

ax.bar3d(
    (xpos).ravel(), ypos.ravel(), np.zeros(dx.size),
    dx.ravel(), dy.ravel(), mmse_err.T.ravel(), color=(.9, .35, .35)
    )

ax.zaxis.set_label_text('Recognition Rate')
ax.xaxis.set_ticks(fdim)
ax.xaxis.set_label_text('Feature Dims')
ax.yaxis.set_label_text('Test Image SNR')
ax.azim = azim
ax.elev = elev
ax.set_zlim(0,1)
ax.set_title('MMSE Recognition under Noise')
f.savefig(
    'dictfigs/mmse_under_noise.pdf', bbox_inches='tight', pad_inches=0
    )

f = pp.figure()
ax = f.add_subplot(111, projection='3d')
ax.bar3d(
    (xpos).ravel(), ypos.ravel(), np.zeros(dx.size),
    dx.ravel(), dy.ravel(), sparse_err.T.ravel(), color=(.9, .35, .35)
    )

ax.zaxis.set_label_text('Recognition Rate')
ax.xaxis.set_ticks(fdim)
ax.xaxis.set_label_text('Feature Dims')
ax.yaxis.set_label_text('Test Image SNR')
ax.azim = azim
ax.elev = elev
ax.set_zlim(0,1)
ax.set_title('Sparse Coding Recognition under Noise')
f.savefig(
    'dictfigs/sparse_under_noise.pdf', bbox_inches='tight', pad_inches=0
    )

pp.show()

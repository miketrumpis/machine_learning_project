from distutils.core import setup, Extension
from numpy.distutils.command import build_src
import Cython
import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = True
from Cython.Distutils import build_ext
import Cython
import numpy

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    numpy_include_dirs = get_numpy_include_dirs()
except AttributeError:
    numpy_include_dirs = numpy.get_include()


dirs = list(numpy_include_dirs) + ['recog']
## dirs.extend(Cython.__path__)
## dirs.append('.')

# C++ Example
## topological = Extension(
##     'video_proj.mean_shift.topological',
##     ['video_proj/mean_shift/topological.pyx'], 
##     include_dirs = dirs,
##     language='c++',
##     extra_compile_args=['-O3']
##     )

# Regular Example
## histogram = Extension(
##     'video_proj.mean_shift.histogram',
##     ['video_proj/mean_shift/histogram.pyx'], 
##     include_dirs = dirs, 
##     extra_compile_args=['-O3']
##     )

if __name__=='__main__':
    setup(
        name = 'recog',
        version = '1.0',
        packages = [
            'recog',
            'recog.conf',
            'recog.faces',
            'recog.dict',
            'recog.opt',
            'recog.support'
            ],
        ext_modules = [],
        package_data = {'recog.conf': ['conf.txt']},
        cmdclass = {'build_ext': build_ext}
    )

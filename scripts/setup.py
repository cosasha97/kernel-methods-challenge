from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension("ssk", ["substring_cython.pyx"])
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)

# command: python setup.py build_ext --inplace

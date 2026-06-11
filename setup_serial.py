# file: setup.py
from setuptools import setup 
from Cython.Build import cythonize
import numpy

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
setup(
    ext_modules = cythonize("process_batch_serial.pyx",compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()],
    
    
)
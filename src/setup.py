from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("cythoncode.cutil", ["cythoncode/cutil.pyx"]),
    Extension("cythoncode.controller_baseline", ["cythoncode/controller_baseline.pyx"]),
    Extension("cythoncode.router_baseline", ["cythoncode/router_baseline.pyx"]),
    Extension("post_processor.box.box_overlaps", ["post_processor/box/box_overlaps.pyx"]),
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[numpy.get_include()]
)

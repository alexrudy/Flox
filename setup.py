# -*- coding: utf-8 -*-
# 
#  setup.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-07.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

import os, os.path
import glob

from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

os.environ["CC"] = "gcc"

# Setup the Cython Extensions
extension_include_dirs = [ np.get_include(), './Flox/']
extension_kwargs = dict(extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'], include_dirs=extension_include_dirs)
extensions = [
    Extension("*", ["Flox/component/*.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/evolver/*.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/finitedifference.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/tridiagonal/*.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/linear/*.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/nonlinear/*.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/magneto/*.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/process/*.pyx"],
        **extension_kwargs),
        ]

# We don't handle dependencies here so that we can use them when running setup.py
DEPENDENCIES = [
]
DEPENDENCY_LINKS = [
]

# Treat everything in scripts except README.rst as a script to be installed
scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
           if os.path.basename(fname) != 'README.rst']
package_info = {}
package_info['options'] = {
      'build_scripts': {
          'executable': 'frpy',
      },}
setup(
    name = 'Flox',
    version = "1.0.0",
    packages = find_packages(),
    install_requires = DEPENDENCIES,
    author = 'Alexander Rudy',
    author_email = 'dev@alexrudy.org',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions),
    scripts = scripts,
    **package_info
)

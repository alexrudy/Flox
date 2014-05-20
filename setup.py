# -*- coding: utf-8 -*-
# 
#  setup.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-07.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

import os

from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

os.environ["CC"] = "gcc"

extension_include_dirs = [ np.get_include(), './Flox/']
extension_kwargs = dict(extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'], include_dirs=extension_include_dirs)
extensions = [
    Extension("*", ["Flox/component/*.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/_threads.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/_solve.pyx"],
        **extension_kwargs),
    Extension("*", ["Flox/_evolve.pyx"],
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
        ]

DEPENDENCIES = [
    # 'astropy',
    # 'pyshell',
    # 'six>=1.4.1',
    # 'numpy>=1.7.1',
    # 'cython'
]
DEPENDENCY_LINKS = [
    ''
]


setup(
    name = 'Flox',
    version = "0.0",
    packages = find_packages(),
    install_requires = DEPENDENCIES,
    author = 'Alexander Rudy',
    author_email = 'dev@alexrudy.org',
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions)
)

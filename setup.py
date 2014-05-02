# -*- coding: utf-8 -*-
# 
#  setup.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-07.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

extension_include_dirs = [ np.get_include(), './Flox/']
extensions = [
    Extension("*", ["Flox/_solve.pyx"],
        include_dirs = extension_include_dirs,),
    Extension("*", ["Flox/finitedifference.pyx"],
        include_dirs = extension_include_dirs,),
    Extension("*", ["Flox/tridiagonal/*.pyx"],
        include_dirs = extension_include_dirs,),
    Extension("*", ["Flox/linear/*.pyx"],
        include_dirs = extension_include_dirs,),
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

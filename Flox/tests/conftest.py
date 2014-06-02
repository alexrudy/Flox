# -*- coding: utf-8 -*-
# 
#  conftest.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-17.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)


import pytest
import numpy as np

from .functional_forms import d_polynomial, d_fourier, FunctionalForm
from .test_system import system_from_args, system_args, system_classes

@pytest.fixture(params=[
    ("Polynomial", 20, 1.0, d_polynomial, (3.0,)),
    ("Polynomial", 20, 1.0, d_polynomial, (3.0,-1.0), True),
    ("Polynomial", 20, 1.0, d_polynomial, (0, 2.0), True),
    ("Polynomial", 20, 1.0, d_polynomial, (0, 2.0), True),
    ("Polynomial", 20, 1.0, d_polynomial, (0, 2.0, 4.0), False),
    ("Polynomial", 20, 1.0, d_polynomial, (0, 2.0, 3.0), False),
    ("Fourier", 20, 1.0, d_fourier, (1.0,), False),
    ("Fourier", 20, 1.0, d_fourier, (1.0,), True),
    ("Fourier", 20, 1.0, d_fourier, (5.0, 3.0, 6.0), False),
])
def functional_form(request):
    """Return a functional form evaluated."""
    return FunctionalForm(*request.param)

@pytest.fixture(params=system_classes)
def system_class(request):
    """Return a system class."""
    return request.param

@pytest.fixture(params=[
    dict(nx=10, nz=10, nn=10),
    dict(nx=100, nz=100, nn=100),
])
def system(request, system_class):
    """Return a system, setup."""
    system_class, this_args = system_class
    this_args.update(system_args)
    return system_from_args(system_class, this_args, request.param)
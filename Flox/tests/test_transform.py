# -*- coding: utf-8 -*-
# 
#  test_transform.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import numpy.random

from Flox.transform import spectral_transform

import pytest

@pytest.fixture(params=[
    (1,1,1), (10,10,10), (400,100,200), (100,100,50)
])
def modal_amplitudes(request):
    """Return modal amplitudes"""
    nz, nm, nx = request.param
    amplitudes = np.random.randn(nz,nm)
    return amplitudes, nx
    
@pytest.mark.smoketest
def test_spectral_transform(modal_amplitudes):
    """Tests a spectral transform against the python implementation."""
    amplitudes, nx = modal_amplitudes
    sa = spectral_transform(np.sin, amplitudes, nx, 1.0, perturbed=False)
    

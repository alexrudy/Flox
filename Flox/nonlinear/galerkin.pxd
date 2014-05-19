# -*- coding: utf-8 -*-
# 
#  galerkin.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange

from Flox._flox cimport DTYPE_t

cpdef int galerkin_sin(int J, int K, DTYPE_t[:,:] G_curr, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] dVdz, DTYPE_t[:,:] O_curr, DTYPE_t[:,:] dOdz, DTYPE_t a, DTYPE_t[:] npa) nogil

cpdef int galerkin_cos(int J, int K, DTYPE_t[:,:] G_curr, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] dVdz, DTYPE_t[:,:] O_curr, DTYPE_t[:,:] dOdz, DTYPE_t a) nogil
# -*- coding: utf-8 -*-
# 
#  currentdensity.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-19.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox.component._solve cimport Solver

cdef class CurrentDensitySolver(Solver):
    cpdef int compute_base(self, DTYPE_t[:,:] A_curr, DTYPE_t[:,:] dAdzz, DTYPE_t[:] npa)
# -*- coding: utf-8 -*-
# 
#  _linear.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox.evolver._evolve cimport Evolver

cdef class HydroEvolver(Evolver):
    cdef public DTYPE_t Pr
    cdef public DTYPE_t Ra
    cdef DTYPE_t maxV
    cdef bint _linear
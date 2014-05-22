# -*- coding: utf-8 -*-
# 
#  _magneto.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-19.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox.evolver._evolve cimport Evolver

cdef class MagnetoEvolver(Evolver):
    cdef public DTYPE_t Pr
    cdef public DTYPE_t Ra
    cdef public DTYPE_t q
    cdef public DTYPE_t Q
    cdef public DTYPE_t maxAlfven
    cdef bint linear_only
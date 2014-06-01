# -*- coding: utf-8 -*-
# 
#  _magneto.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-19.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox.evolver._hydro cimport HydroEvolver

cdef class MagnetoEvolver(HydroEvolver):
    cdef public DTYPE_t q
    cdef public DTYPE_t Q
    cdef public DTYPE_t maxAlfven

# -*- coding: utf-8 -*-
# 
#  _linear.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox._evolve cimport Evolver
    
cdef class LinearEvolver(Evolver):
    cdef public DTYPE_t Pr
    cdef public DTYPE_t Ra

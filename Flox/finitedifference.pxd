# -*- coding: utf-8 -*-
# 
#  finitedifference.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 University of California. All rights reserved.
# 

from Flox._flox cimport DTYPE_t

cpdef int second_derivative2D(int J, int K, DTYPE_t[:,:] ddf, DTYPE_t[:,:] f, DTYPE_t dz, DTYPE_t f_p, DTYPE_t f_m, DTYPE_t factor)

cpdef int second_derivative(int J, DTYPE_t[:] ddf, DTYPE_t[:] f, DTYPE_t dz, DTYPE_t f_p, DTYPE_t f_m, DTYPE_t factor)
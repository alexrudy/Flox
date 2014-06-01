# -*- coding: utf-8 -*-
# 
#  _transform.pxd
#  flox
#  
#  Created by Alexander Rudy on 2014-05-20.
#  Copyright 2014 University of California. All rights reserved.
# 

from Flox._flox cimport DTYPE_t

cpdef int transform(int J, int K, int Kx, DTYPE_t[:,:] V_trans, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] _transform) nogil


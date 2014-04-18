# 
#  _tridiagonal.pyd
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 University of California. All rights reserved.
# 

from Flox._flox cimport DTYPE_t

cpdef int tridiagonal_split_matrix(DTYPE_t[:,:] mat, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup)

cpdef int tridiagonal_from_matrix(DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:,:] mat)

cpdef int tridiagonal_solver(DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup)

cpdef int tridiagonal_do_work(DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup, DTYPE_t[:] wk1, DTYPE_t[:] wk2)

cpdef int tridiagonal_from_work(DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:] wk1, DTYPE_t[:] wk2, DTYPE_t[:] sub)
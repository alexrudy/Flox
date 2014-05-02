# 
#  _tridiagonal.pyd
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 University of California. All rights reserved.
# 

from Flox._flox cimport DTYPE_t

cpdef int tridiagonal_split_matrix(int J, DTYPE_t[:,:] mat, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup)

cpdef int tridiagonal_from_matrix(int J, DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:,:] mat)

cpdef int tridiagonal_solver(int J, DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup)

cpdef int tridiagonal_do_work(int J, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup, DTYPE_t[:] wk1, DTYPE_t[:] wk2)

cpdef int tridiagonal_from_work(int J, DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:] wk1, DTYPE_t[:] wk2, DTYPE_t[:] sub)

cdef class TridiagonalSolver:
    cdef DTYPE_t[:,:] wk1
    cdef DTYPE_t[:,:] wk2
    cdef DTYPE_t[:,:] sub
    cdef DTYPE_t[:,:] dia
    cdef DTYPE_t[:,:] sup
    cdef readonly int J
    cdef readonly int K
    cdef bint warmed
    cdef int _warm_work(self)
    cpdef int solve(self, DTYPE_t[:,:] rhs, DTYPE_t[:,:] sol)
    cpdef int warm(self, DTYPE_t[:,:] sub, DTYPE_t[:,:] dia, DTYPE_t[:,:] sup)
    cpdef int matrix(self, DTYPE_t[:,:,:] mat)


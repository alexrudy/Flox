# -*- coding: utf-8 -*-
# 
#  _tridiagonal.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-17.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

# Look into array allocation here:
# http://stackoverflow.com/questions/18462785/what-is-the-recommended-way-of-allocating-memory-for-a-typed-memory-view

#cython: overflowcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t

cpdef int tridiagonal_split_matrix(DTYPE_t[:,:] mat, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup):
    
    cdef int j, J = mat.shape[0]

    sup[0] = mat[0,1]
    dia[0] = mat[0,0]
    for j in range(1, J-1):
        sup[j] = mat[j,j+1]
        dia[j] = mat[j,j]
        sub[j] = mat[j,j-1]
    dia[J-1] = mat[J-1,J-1]
    sub[J-1] = mat[J-1,J-2]
    
    return 0

cpdef int tridiagonal_from_matrix(DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:,:] mat):
    
    cdef int r1, r2, J = rhs.shape[0]
    cdef DTYPE_t[:] sub = clone(array('d'), J, False)
    cdef DTYPE_t[:] dia = clone(array('d'), J, False)
    cdef DTYPE_t[:] sup = clone(array('d'), J, False)
    
    r1 = tridiagonal_split_matrix(mat, sub, dia, sup)
    r2 = tridiagonal_solver(rhs, sol, sub, dia, sup)
    
    return (r1 + r2)
    

cpdef int tridiagonal_solver(DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup):
    
    cdef int r1, r2, j, J = rhs.shape[0]
    cdef DTYPE_t[:] wk1 = clone(array('d'), J, False)
    cdef DTYPE_t[:] wk2 = clone(array('d'), J, False)
    
    r1 = tridiagonal_do_work(sub, dia, sup, wk1, wk2)
    r2 = tridiagonal_from_work(rhs, sol, wk1, wk2, sub)
    
    return r1 + r2

cpdef int tridiagonal_do_work(DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup, DTYPE_t[:] wk1, DTYPE_t[:] wk2):
    
    cdef int j, J = sub.shape[0]
    
    wk1[0] = 1.0 / dia[0]
    wk2[0] = sup[0] * wk1[0]
    for j in range(1, J-1):
        wk1[j] = 1.0 / (dia[j] - sub[j] * wk2[j-1])
        wk2[j] = sup[j] * wk1[j]
    
    wk1[J-1] = 1.0 / (dia[J-1] - sub[J-1] * wk2[J-2])
    
    return 0

cpdef int tridiagonal_from_work(DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:] wk1, DTYPE_t[:] wk2, DTYPE_t[:] sub):
    
    cdef int j, J = rhs.shape[0]
    
    sol[0] = rhs[0] * wk1[0]
    for j in range(1, J):
        sol[j] = (rhs[j] - sub[j] * sol[j-1]) * wk1[j]
    
    for j in range(J-2, -1, -1):
        sol[j] = sol[j] - wk2[j] * sol[j+1]
        
    return 0
    

cdef class TridiagonalSolver:
    
    # cdef DTYPE_t[:] wk1
    # cdef DTYPE_t[:] wk2
    # cdef DTYPE_t[:] sub
    # cdef DTYPE_t[:] dia
    # cdef DTYPE_t[:] sup
    # cdef int J
    # cdef bint warmed
    
    def __cinit__(self, int size):
        
        self.J = size
        self.warmed = False
        self.wk1 = clone(array('d'), self.J, False)
        self.wk2 = clone(array('d'), self.J, False)
        self.sub = clone(array('d'), self.J, False)
        self.dia = clone(array('d'), self.J, False)
        self.sup = clone(array('d'), self.J, False)
        
    
    cpdef int warm(self, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup):
        self.sub = sub
        self.dia = dia
        self.sub = sub
        return self._warm_work()
    
    cdef _warm_work(self):
        return tridiagonal_do_work(self.sub, self.dia, self.sup, self.wk1, self.wk2)
    
    cpdef int solve(self, DTYPE_t[:] rhs, DTYPE_t[:] sol):
        return tridiagonal_from_work(rhs, sol, self.wk1, self.wk2, self.sub)
    
    cpdef int matrix(self, DTYPE_t[:,:] mat):
        cdef int r1, r2
        r1 = tridiagonal_split_matrix(mat, self.sub, self.dia, self.sup)
        r2 = self._warm_work()
        return r1 + r2
        
    
    
    
    

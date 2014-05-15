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

#cython: overflowcheck=True
#cython: wraparound=False
#cython: boundscheck=True
#cython: cdivision=True
#cython: profile=True

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t
from Flox._solve cimport Solver

cpdef int tridiagonal_split_matrix(int J, DTYPE_t[:,:] mat, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup):
    
    cdef int j
    
    j = 0
    sub[j] = 0.0
    sup[j] = 0.0
    dia[j] = 0.0
    j = 1
    sub[j] = 0.0
    sup[j] = mat[j-1,j]
    dia[j] = mat[j-1,j-1]
    for j in range(2, J-2):
        sup[j] = mat[j-1,j]
        dia[j] = mat[j-1,j-1]
        sub[j] = mat[j-1,j-2]
    j = J-2
    sup[j] = 0.0
    dia[j] = mat[j-1,j-1]
    sub[j] = mat[j-1,j-2]
    
    dia[J-1] = 0.0
    sub[J-1] = 0.0
    sup[J-1] = 0.0
    
    return 0

cpdef int tridiagonal_from_matrix(int J, DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:,:] mat):
    
    cdef int r1, r2
    cdef DTYPE_t[:] sub = clone(array('d'), J, False)
    cdef DTYPE_t[:] dia = clone(array('d'), J, False)
    cdef DTYPE_t[:] sup = clone(array('d'), J, False)
    
    r1 = tridiagonal_split_matrix(J, mat, sub, dia, sup)
    r2 = tridiagonal_solver(J, rhs, sol, sub, dia, sup)
    
    return (r1 + r2)
    

cpdef int tridiagonal_solver(int J, DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup):
    
    cdef int r1, r2
    cdef DTYPE_t[:] wk1 = clone(array('d'), J, False)
    cdef DTYPE_t[:] wk2 = clone(array('d'), J, False)
    
    r1 = tridiagonal_do_work(J, sub, dia, sup, wk1, wk2)
    r2 = tridiagonal_from_work(J, rhs, sol, wk1, wk2, sub)
    
    return r1 + r2

cpdef int tridiagonal_do_work(int J, DTYPE_t[:] sub, DTYPE_t[:] dia, DTYPE_t[:] sup, DTYPE_t[:] wk1, DTYPE_t[:] wk2):
    
    cdef int j
    
    wk1[0] = 1.0 / dia[0]
    wk2[0] = sup[0] * wk1[0]
    for j in range(1, J-1):
        wk1[j] = 1.0 / (dia[j] - sub[j] * wk2[j-1])
        wk2[j] = sup[j] * wk1[j]
    
    wk1[J-1] = 1.0 / (dia[J-1] - sub[J-1] * wk2[J-2])
    
    return 0

cpdef int tridiagonal_from_work(int J, DTYPE_t[:] rhs, DTYPE_t[:] sol, DTYPE_t[:] wk1, DTYPE_t[:] wk2, DTYPE_t[:] sub):
    
    cdef int j
    
    j = 0
    sol[j] = rhs[j] * wk1[j]
    for j in range(1, J):
        sol[j] = (rhs[j] - sub[j] * sol[j-1]) * wk1[j]
        
    for j in range(J-2, -1, -1):
        sol[j] = sol[j] - wk2[j] * sol[j+1]
    return 0
    

cdef class TridiagonalSolver(Solver):
    
    def __cinit__(self, int nz, int nx):
        
        self.J = nz + 2
        self.K = nx
        self.warmed = False
        self.wk1 = np.zeros((self.J, self.K), dtype=np.float)
        self.wk2 = np.zeros((self.J, self.K), dtype=np.float)
        self.sub = np.zeros((self.J, self.K), dtype=np.float)
        self.dia = np.zeros((self.J, self.K), dtype=np.float)
        self.sup = np.zeros((self.J, self.K), dtype=np.float)
        self.t_sol = np.zeros((self.J,), dtype=np.float)
        self.t_rhs = np.zeros((self.J,), dtype=np.float)
        
    
    cpdef int warm(self, DTYPE_t[:,:] sub, DTYPE_t[:,:] dia, DTYPE_t[:,:] sup):
        self.sub = sub
        self.dia = dia
        self.sub = sub
        return self._warm_work()
    
    cdef int _warm_work(self):
        
        cdef int k, rv = 0
        cdef DTYPE_t[:] t_wk1 = clone(array('d'), self.J, False)
        cdef DTYPE_t[:] t_wk2 = clone(array('d'), self.J, False)
        
        for k in range(self.K):
            rv += tridiagonal_do_work(self.J, self.sub[:,k], self.dia[:,k], self.sup[:,k], t_wk1, t_wk2)
            self.wk1[:,k] = t_wk1
            self.wk2[:,k] = t_wk2
        
        self.warmed = True
        
        return rv
    
    cpdef int solve(self, DTYPE_t[:,:] rhs, DTYPE_t[:,:] sol):
        
        cdef int k, rv = 0
        
        for k in range(self.K):
            self.t_rhs[1:self.J-1] = rhs[:,k]
            self.t_sol[:] = 0.0
            rv += tridiagonal_from_work(self.J, self.t_rhs, self.t_sol, self.wk1[:,k], self.wk2[:,k], self.sub[:,k])
            sol[:,k] = self.t_sol[1:self.J-1]
        
        return rv
    
    cpdef int matrix(self, DTYPE_t[:,:,:] mat):
        cdef int r1 = 0, r2, k
        cdef DTYPE_t[:] t_sub = clone(array('d'), self.J, False)
        cdef DTYPE_t[:] t_dia = clone(array('d'), self.J, False)
        cdef DTYPE_t[:] t_sup = clone(array('d'), self.J, False)

        for k in range(self.K):
            r1 += tridiagonal_split_matrix(self.J, mat[:,:,k], t_sub, t_dia, t_sup)
            self.sub[:,k] = t_sub
            self.dia[:,k] = t_dia
            self.sup[:,k] = t_sup
        
        r2 = self._warm_work()
        return r1 + r2
        
    
    
    
    

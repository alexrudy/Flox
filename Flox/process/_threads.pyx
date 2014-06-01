# -*- coding: utf-8 -*-
# 
#  _threads.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-22.
#  Copyright 2014 University of California. All rights reserved.
# 

cimport cython
cimport openmp

cpdef int omp_get_num_threads():
    return openmp.omp_get_num_threads()

cpdef int omp_set_num_threads(int threads):
    openmp.omp_set_num_threads(threads)
    return 0
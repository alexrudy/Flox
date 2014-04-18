#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
#  tridiagonal.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-17.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import timeit
import textwrap
import numpy as np
import os, os.path

import matplotlib.pyplot as plt

def tridiagonal_trial_setup(n, seed):
    """Do a single trial with a n=? trial."""
    from Flox.tridiagonal import tridiagonal_from_matrix
    from Flox.tridiagonal.test_tridiagonal import assemble_tridiagonal_matrix, assemble_solution_matrix
    n = int(n)
    eps = 1e-3
    
    mat = assemble_tridiagonal_matrix(n, eps, seed=seed)
    sol = assemble_solution_matrix(n, seed=seed)
    
    rhs = np.array(mat * sol)[:,0]
    res = np.array(np.zeros_like(sol))[:,0]
    
    return rhs, res, mat
    
def tridiagonal_split_trial_setup(n, seed):
    """Test pre-split arrays."""
    from Flox.tridiagonal import tridiagonal_split_matrix
    from Flox.tridiagonal.test_tridiagonal import assemble_tridiagonal_matrix, assemble_solution_matrix
    n = int(n)
    eps = 1e-3
    
    mat = assemble_tridiagonal_matrix(n, eps, seed=seed)
    sol = assemble_solution_matrix(n, seed=seed)
    
    rhs = np.array(mat * sol)[:,0]
    res = np.array(np.zeros_like(sol))[:,0]
    sub = np.zeros_like(res)
    dia = np.zeros_like(res)
    sup = np.zeros_like(res)
    
    status = tridiagonal_split_matrix(mat, sub, dia, sup)
    
    return rhs, res, sub, dia, sup

def tridiagonal_trial_setup_cmd(n, seed):
    """Setup command"""
    return textwrap.dedent("""
    from __main__ import tridiagonal_trial_setup
    from Flox.tridiagonal import tridiagonal_from_matrix
    rhs, res, mat = tridiagonal_trial_setup({n:d},{seed:d})    
    """.format(n=n,seed=seed))
    
def tridiagonal_split_trial_setup_cmd(n, seed):
    """Setup command"""
    return textwrap.dedent("""
    from __main__ import tridiagonal_split_trial_setup
    from Flox.tridiagonal import tridiagonal_solver
    rhs, res, sup, dia, sub = tridiagonal_split_trial_setup({n:d},{seed:d})    
    """.format(n=n,seed=seed))

def tridiagonal_trial(n, seed):
    """Tridiagonal trial."""
    cmd = "status = tridiagonal_from_matrix(rhs, res, mat)"
    print("Trying n={:d}".format(n))
    return timeit.timeit(cmd, setup=tridiagonal_trial_setup_cmd(n, seed), number=int(1e6))

def tridiagonal_split_trial(n, seed):
    """Tridiagonal trial."""
    cmd = "status = tridiagonal_solver(rhs, res, sub, dia, sup)"
    print("Trying n={:d}".format(n))
    return timeit.timeit(cmd, setup=tridiagonal_split_trial_setup_cmd(n, seed), number=int(1e6))


def tridiagonal_timing(ns, seed):
    """Do the timing trials"""
    return [tridiagonal_trial(n, seed) for n in ns]

if __name__ == '__main__':
    
    filename = os.path.join(os.path.dirname(__file__),"tridiagonal.npy")
    if os.path.exists(filename):
        ns, times, times_split = np.hsplit(np.load(filename).T, 3)
    else:
        ns = np.power(2, np.arange(1,10))
        times_split = [tridiagonal_split_trial(n, 5) for n in ns]
        times = tridiagonal_timing(ns, 5)
        np.save(filename, np.vstack((ns, times, times_split)))
    
    print("Plotting Timing Results")
    plotname = os.path.join(os.path.dirname(__file__),"tridiagonal.pdf")
    plt.plot(ns, times, 'bo-', label="With Matrix Split")
    plt.plot(ns, times_split, 'go-', label="Pre-split arrays")
    plt.xlabel(r"Matrix Size $(n \times n)$")
    plt.ylabel(r"Time per loop $(\mu s)$")
    plt.title(r"Tridiagonal Algorithm Timing")
    plt.legend(loc="upper left")
    plt.savefig(plotname)
    

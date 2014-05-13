# -*- coding: utf-8 -*-
# 
#  plot.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-10.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

"""
Standard plotting tools for the nonlinear code.
"""

from Flox.plot import GridView, ContourView, MultiViewController, EvolutionViewStabilityTest, EvolutionViewSingleMode

def setup_plots(fig, stability=None, zmode=33):
    """Setup plots from a figure."""
    rows = 3 if stability is not None else 1
    MVC = MultiViewController(fig, rows, 3)
    MVC[0,0] = GridView("Temperature", cmap="hot")
    MVC[0,1] = GridView("Vorticity", cmap='Blues')
    MVC[0,2] = ContourView("Stream", cmap='Greens')
    if stability is not None:
        MVC[1,0] = EvolutionViewSingleMode("Temperature", stability, zmode)
        MVC[1,1] = EvolutionViewSingleMode("Vorticity", stability, zmode)
        MVC[1,2] = EvolutionViewSingleMode("Stream", stability, zmode)
        MVC[2,0] = EvolutionViewSingleMode("dTemperature", stability, zmode)
        MVC[2,1] = EvolutionViewSingleMode("dVorticity", stability, zmode)
    return MVC
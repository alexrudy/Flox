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
# from maptloplib.colors import LogNorm
from Flox.plot import GridView, ContourView, MultiViewController, EvolutionViewStabilityTest, EvolutionViewSingleMode, RawGridView, VProfileView, V2DProfileView, V1DProfileView

def setup_plots(fig, stability=None, zmode=33):
    """Setup plots from a figure."""
    rows = 4 if stability is not None else 2
    MVC = MultiViewController(fig, rows, 3)
    MVC[0,0] = GridView("Temperature", cmap="hot")
    MVC[0,1] = GridView("Vorticity", cmap='Blues')
    MVC[0,2] = GridView("Stream", cmap='Greens')
    MVC[1,0] = RawGridView("Temperature", cmap="hot")
    MVC[1,1] = RawGridView("Vorticity", cmap='Blues')
    MVC[1,2] = RawGridView("Stream", cmap='Greens')
    if stability is not None:
        MVC[2,0] = EvolutionViewSingleMode("Temperature", stability, zmode)
        MVC[2,1] = EvolutionViewSingleMode("Vorticity", stability, zmode)
        MVC[2,2] = EvolutionViewSingleMode("Stream", stability, zmode)
        MVC[3,0] = EvolutionViewSingleMode("dTemperature", stability, zmode)
        MVC[3,1] = EvolutionViewSingleMode("dVorticity", stability, zmode)
        MVC[3,2] = VProfileView("Temperature", 0)
    return MVC
    
def setup_movie(fig, variables=["Temperature"], kwargs=[dict(cmap="hot")]):
    """Setup a movie view."""
    MVC = MultiViewController(fig, len(variables), 1)
    for i, (variable, kwds) in enumerate(zip(variables,kwargs)):
        MVC[i,0] = GridView(variable,**kwds)
    return MVC
    

def setup_plots_watch(fig, stability=None, zmode=33):
    """Setup plots from a figure."""
    rows = 5
    MVC = MultiViewController(fig, rows, 3)
    MVC[0,0] = GridView("Temperature", cmap="hot")
    MVC[0,1] = GridView("Vorticity", cmap='Blues')
    MVC[0,2] = GridView("Stream", cmap='Greens')
    MVC[1,0] = RawGridView("Temperature", cmap="hot")
    MVC[1,1] = RawGridView("Vorticity", cmap='Blues')
    MVC[1,2] = RawGridView("Stream", cmap='Greens')
    MVC[2,0] = VProfileView("Temperature", stability)
    MVC[2,1] = VProfileView("Vorticity", stability)
    MVC[2,2] = VProfileView("Stream", stability)
    MVC[3,0] = V2DProfileView("Temperature", stability)
    MVC[3,1] = V2DProfileView("Vorticity", stability)
    MVC[3,2] = V2DProfileView("Stream", stability)
    MVC[4,0] = V1DProfileView("Temperature", stability)
    MVC[4,1] = V1DProfileView("Vorticity", stability)
    MVC[4,2] = V1DProfileView("Stream", stability)
    return MVC
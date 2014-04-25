# -*- coding: utf-8 -*-
# 
#  plot.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-23.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from matplotlib.gridspec import GridSpec
import numpy as np

class MultiViewController(object):
    """A controller which manages many views."""
    def __init__(self, figure, nr, nc):
        super(MultiViewController, self).__init__()
        self.figure = figure
        self.gs = GridSpec(nr, nc)
        self.views = []
        
    def __setitem__(self, key, view):
        """Set the slice item in this figure."""
        view.ax = self.figure.add_subplot(self.gs[key])
        self.views.append(view)
        
    def update(self, system):
        """Update all plots with the given system."""
        for view in self.views:
            view.update(system)
        self.draw()
        
    def draw(self):
        """Force the figure to re-draw."""
        self.figure.canvas.draw()

class EvolutionView(object):
    """An object view showing the time-evolution of a parameter."""
    def __init__(self, variable):
        super(EvolutionView, self).__init__()
        self.variable = variable
        self.ax = None
        self.line = None
    
    def ydata(self, system):
        """Return the y-data values."""
        return getattr(system, self.variable)[...,:system.it]
    
    def initialize(self, system):
        """Set up the plot."""
        self.line, = self.ax.plot(system.Time[:system.it], self.ydata(system), 'k-')
        self.ax.set_ylabel(getattr(type(system), self.variable).latex)
        self.ax.set_xlabel(type(system).Time.latex)
        
    def update(self, system):
        """Update the view."""
        if self.line is None:
            self.initialize(system)
        self.line.set_data(system.Time[:system.it], self.ydata(system))
        
    
class EvolutionViewAllModes(EvolutionView):
    """Watch a variable evolve for all fourier modes."""
    
    def initialize(self, system):
        """Set up the plot."""
        self.line = []
        for i in range(self.ydata(system).shape[1]):
            self.line.append(self.ax.plot(system.Time[:system.it], np.mean(self.ydata(system)[:,i,:], axis=0), 'k-')[0])
        self.ax.set_ylabel(getattr(type(system), self.variable).latex)
        self.ax.set_xlabel(type(system).Time.latex)
    
    def update(self, system):
        """Update the view."""
        if self.line is None:
            self.initialize(system)
        for i, line in enumerate(self.line):
            line.set_data(system.Time[:system.it], np.mean(self.ydata(system)[:,i,:], axis=0))
    
class EvolutionViewSingleMode(EvolutionView):
    """Watch a variable evolve for a single fourier mode."""
    
    def __init__(self, variable, nmode):
        super(EvolutionViewSingleMode, self).__init__(variable)
        self.nmode = nmode
    
    def ydata(self, system):
        """Return the y-data values."""
        return np.mean(getattr(system, self.variable)[:,self.nmode,:system.it], axis=0)
        



        
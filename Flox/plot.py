# -*- coding: utf-8 -*-
# 
#  plot.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-23.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import abc
import six
from matplotlib.gridspec import GridSpec
import numpy as np

class MultiViewController(object):
    """A controller which manages many views."""
    def __init__(self, figure, nr, nc, **kwargs):
        super(MultiViewController, self).__init__()
        self.figure = figure
        self.gs = GridSpec(nr, nc, **kwargs)
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

@six.add_metaclass(abc.ABCMeta)
class View(object):
    """A view controller"""
    
    @abc.abstractmethod
    def update(self, system):
        """Update this view."""
        pass

class GridView(View):
    """View this object on a grid."""
    def __init__(self, variable, perturbed=False, **kwargs):
        super(GridView, self).__init__()
        self.variable = variable
        self.ax = None
        self.image = None
        self.im_kwargs = kwargs
        self.perturbed = perturbed
        
    def data(self, system):
        """Return the transformed data"""
        return system.transformed_array(self.variable, perturbed=self.perturbed)
        
    def initialize(self, system):
        """Initialize the system."""
        self.im_kwargs.setdefault('cmap','hot')
        self.im_kwargs['aspect'] = 1.0 / system.aspect * (system.nx / system.nz)
        self.image = self.ax.imshow(self.data(system).value, **self.im_kwargs)
        self.ax.figure.colorbar(self.image, ax=self.ax)
        self.title = self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.counter = self.ax.text(0.05, 1.15, "t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.it, system.nit), transform=self.ax.transAxes)
        
    def update(self, system):
        """Update the view"""
        if self.image is None:
            self.initialize(system)
        else:
            self.image.set_data(self.data(system).value)
            self.image.autoscale()
            self.counter.set_text("t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.it, system.nit))


class ContourView(GridView):
    """Show countours."""
    
    def initialize(self, system):
        """Initialize the system."""
        self.im_kwargs.setdefault('cmap','hot')
        self.im_kwargs['aspect'] = 1.0 / system.aspect * (system.nx / system.nz)
        self.image = self.ax.contour(self.data(system).value, **self.im_kwargs)
        self.title = self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.counter = self.ax.text(0.05, 1.15, "t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.it, system.nit), transform=self.ax.transAxes)
        
    def update(self, system):
        """Update the view"""
        if self.image is None:
            self.initialize(system)
        else:
            self.image.set_data(self.data(system).value)
            self.counter.set_text("t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.it, system.nit))
    

class EvolutionView(View):
    """An object view showing the time-evolution of a parameter."""
    def __init__(self, variable):
        super(EvolutionView, self).__init__()
        self.variable = variable
        self.ax = None
        self.line = None
    
    def ydata(self, system):
        """Return the y-data values."""
        return system.engine[self.variable][...,:system.it]
        
    def xdata(self, system):
        """Return the x-data values."""
        return system.dimensional_full_array("Time")[:system.it]
    
    def initialize(self, system):
        """Set up the plot."""
        self.line, = self.ax.plot(self.xdata(system), self.ydata(system), 'k-')
        self.ax.set_ylabel(getattr(type(system), self.variable).latex)
        self.ax.set_xlabel(type(system).Time.latex)
        self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        
        
    def update(self, system):
        """Update the view."""
        if self.line is None:
            self.initialize(system)
        self.line.set_data(self.xdata(system), self.ydata(system))
        self.ax.relim()
        self.ax.autoscale_view()
        
    
class EvolutionViewAllModes(EvolutionView):
    """Watch a variable evolve for all fourier modes."""
    
    def __init__(self, variable, zmode):
        super(EvolutionViewAllModes, self).__init__(variable)
        self.zmode = zmode
        
    
    def initialize(self, system):
        """Set up the plot."""
        self.line = []
        for i in range(self.ydata(system).shape[1]):
            self.line.append(self.ax.plot(self.xdata(system), self.ydata(system)[self.zmode,i,:], 'k-')[0])
        self.ax.set_ylabel(getattr(type(system), self.variable).latex)
        self.ax.set_xlabel(type(system).Time.latex)
        self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
    
    def update(self, system):
        """Update the view."""
        if self.line is None:
            self.initialize(system)
        for i, line in enumerate(self.line):
            line.set_data(self.xdata(system), self.ydata(system)[self.zmode,i,:])
    
class EvolutionViewSingleMode(EvolutionView):
    """Watch a variable evolve for a single fourier mode."""
    
    def __init__(self, variable, nmode, zmode):
        super(EvolutionViewSingleMode, self).__init__(variable)
        self.nmode = nmode
        self.zmode = zmode
    
    def initialize(self, system):
        """Setup a single mode"""
        super(EvolutionViewSingleMode, self).initialize(system)
        self.ax.title.set_text("[N={}, z={}] {} ({})".format(self.nmode, self.zmode, getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
    
    def ydata(self, system):
        """Return the y-data values."""
        return system.engine[self.variable][self.zmode,self.nmode,:system.it]
        

class EvolutionViewStabilityTest(EvolutionViewSingleMode):
    """Show a varaible's stability test parameter."""
        
    def update(self, system):
        if system.it <= 1:
            return
        super(EvolutionViewStabilityTest, self).update(system)
        
    def initialize(self, system):
        """Setup the y-axis label"""
        super(EvolutionViewStabilityTest, self).initialize(system)
        latex = getattr(type(system), self.variable).latex[1:-1]
        self.ax.yaxis.label.set_text(r"$\ln({latex:s}_t) - \ln({latex:s}_{{t-1}})$".format(latex=latex))
        
        ydata = self.ydata(system)
        if len(ydata):
            value = ydata[-1]
        else:
            value = np.nan
        self.label = self.ax.text(0.95, 0.95, "{:.2g}".format(value), transform=self.ax.transAxes, va='top', ha='right')
        
    def update(self, system):
        """Update."""
        super(EvolutionViewStabilityTest, self).update(system)
        ydata = self.ydata(system)
        if len(ydata):
            value = ydata[-1]
        else:
            value = np.nan
        self.label.set_text("{:.2g}".format(value))
        
    def ydata(self, system):
        """Return the y-data values."""
        data = system.engine[self.variable][self.zmode,self.nmode,:system.it]
        ln_data = np.log(np.abs(data))
        return np.diff(ln_data)
        
    def xdata(self, system):
        """Return the x-data values."""
        return system.dimensional_full_array("Time")[1:system.it]
        

        
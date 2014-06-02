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
import sys
import io
import six
from matplotlib.gridspec import GridSpec
import numpy as np
import queue
import warnings

from astropy.utils.console import ProgressBar
from matplotlib import animation

from pyshell.util import configure_class

from .util import callback_progressbar_wrapper

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
        
    @classmethod
    def from_config(cls, config):
        """Configure this object."""
        import matplotlib.pyplot as plt
        figure = plt.figure(**config.get('fig',{}))
        mvc = cls(figure, config.get('nr',1), config.get('nc', 1), **config.get('kwargs',{}))
        for plot in config['plots'].values():
            r, c = plot.pop('r'), plot.pop('c')
            mvc[r,c] = configure_class(plot)
        return mvc
        
    def get_animation(self, system, queue, progressbar=None, buffer=10, timeout=60, **kwargs):
        """Get the animation object."""
        generator = self._animation_generator(system, queue=queue, buffer=buffer, timeout=timeout)
        callback = self._animation_callback(progressbar=progressbar)
        # kwargs.setdefault('repeat', False)
        kwargs.setdefault('save_count', int(system.engine.iterations))
        return animation.FuncAnimation(self.figure, func=callback, frames=generator, init_func=lambda : self.update(system), **kwargs)
        
    def _animation_generator(self, system, queue=None, buffer=10, timeout=60):
        """Get the animation generator."""
        if queue is not None:
            generator = lambda : system.iterate_queue_buffered(queue, buffer=buffer, timeout=timeout)
        else:
            generator = lambda : iter(system)
        return generator
    
    def _animation_callback(self, progressbar=None):
        """Get the animator callback."""
        if progressbar is None:
            callback = self.update
        else:
            # callback_progressbar_wrapper expects arguments of the form iteration, *args
            # We do this trick to insert the iterations into the arglist, then remove them.
            _callback = callback_progressbar_wrapper(lambda i, s : self.update(s), progressbar)
            callback = lambda s : _callback(s.iteration, s)
        return callback
        
    def animate(self, system, queue=None, progress=True, **kwargs):
        """Animation."""
        import matplotlib
        import matplotlib.pyplot as plt
        
        # Disable LaTeX while animating!
        matplotlib.rcParams['text.usetex'] = False
        
        out = None if progress else io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ProgressBar(system.engine.iterations, file=out) as progressbar:
                anim = self.get_animation(system, queue, progressbar, **kwargs)
                plt.show()
            
        
    def movie(self, filename, system, queue=None, progress=True, save_kwargs=dict(), **kwargs):
        """Write a movie."""
        import matplotlib
        import matplotlib.pyplot as plt
        
        # Disable LaTeX while animating!
        matplotlib.rcParams['text.usetex'] = False
        
        out = None if progress else io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ProgressBar(system.engine.iterations, file=out) as PBar:
                anim = self.get_animation(system, queue, PBar, buffer=0, **kwargs)
                anim.save(filename, **save_kwargs)

@six.add_metaclass(abc.ABCMeta)
class View(object):
    """A view controller"""
    
    def __init__(self):
        """Initializer."""
        self.ax = None
        self.initialized = False
        
    @abc.abstractmethod
    def initialize(self, system):
        """Initialize the view."""
        self.initialized = True
    
    @abc.abstractmethod
    def update(self, system):
        """Update this view."""
        if not self.initialized:
            self.initialize(system)
        

class GridView(View):
    """View this object on a grid."""
    def __init__(self, variable, perturbed=False, **kwargs):
        super(GridView, self).__init__()
        self.variable = variable
        self.image = None
        self.im_kwargs = kwargs
        self.perturbed = perturbed
        
    def data(self, system):
        """Return the transformed data"""
        if self.perturbed:
            data = getattr(system,self.variable).perturbation.value
        else:
            data = getattr(system,self.variable).transformed.value
        if self.variable == "VectorPotential":
            return data + system.B0.value * np.linspace(0, 1, system.nx + 2)[1:-1][np.newaxis,:]
        else:
            return data
        
    def initialize(self, system):
        """Initialize the system."""
        super(GridView, self).initialize(system)
        self.im_kwargs.setdefault('cmap','hot')
        self.im_kwargs['aspect'] = 1.0 / system.aspect * (system.nx / system.nz)
        self.image = self.ax.imshow(self.data(system), **self.im_kwargs)
        self.ax.figure.colorbar(self.image, ax=self.ax)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.title = self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.counter = self.ax.text(0.05, 1.15, "t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.iteration, system.engine.iterations), transform=self.ax.transAxes)
        
    def update(self, system):
        """Update the view"""
        super(GridView, self).update(system)
        self.image.set_data(self.data(system))
        self.image.autoscale()
        self.counter.set_text("t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.iteration, system.engine.iterations))


class RawGridView(GridView):
    """RawGridView"""
    
    def data(self, system):
        """Return the transformed data"""
        return getattr(system, self.variable).raw

class ContourView(GridView):
    """Show countours."""
    
    def initialize(self, system):
        """Initialize the system."""
        aspect = 1.0 / system.aspect * (system.nx / system.nz)
        data = self.data(system)
        ptp = np.ptp(data)
        if np.isfinite(ptp) and ptp != 0.0:
            self.image = self.ax.contour(self.data(system), 10, **self.im_kwargs)
            self.cb = self.ax.figure.colorbar(self.image, ax=self.ax)
        self.ax.set_aspect(aspect)
        self.title = self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.counter = self.ax.text(0.05, 1.15, "t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.iteration, system.engine.iterations), transform=self.ax.transAxes)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.initialized = True
        
    def update(self, system):
        """Update the view"""
        if not self.initialized:
            self.initialize(system)
        aspect = 1.0 / system.aspect * (system.nx / system.nz)
        self.ax.cla()
        self.ax.set_aspect(aspect)
        data = self.data(system)
        ptp = np.ptp(data)
        if np.isfinite(ptp) and ptp != 0.0:
            self.image = self.ax.contour(self.data(system), 10, **self.im_kwargs)
            if hasattr(self, 'cb'):
                self.cb.ax.clear()
                self.cb = self.ax.figure.colorbar(self.image, ax=self.ax, cax=self.cb.ax)
            else:
                self.cb = self.ax.figure.colorbar(self.image, ax=self.ax)
        self.title = self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.counter.set_text("t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.iteration, system.engine.iterations))
    

class VectorView(GridView):
    """Vector field view."""
    
    def data(self, system):
        """docstring for data"""
        return getattr(system,self.variable).value
    
    def initialize(self, system):
        """Initialize the system."""
        aspect = 1.0 / system.aspect * (system.nx / system.nz)
        data = self.data(system)
        ptp = np.ptp(data)
        if np.isfinite(ptp) and ptp != 0.0:
            z,x = np.mgrid[:system.nz,:system.nn]
            u,v = data
            speed = np.sqrt(u**2.0 + v**2.0)
            self.image = self.ax.streamplot(x,z,u,v,color=speed, **self.im_kwargs)
            self.cb = self.ax.figure.colorbar(self.image.lines, ax=self.ax)
        self.ax.set_xlim(0, system.nx)
        self.ax.set_ylim(0, system.nz)
        self.ax.set_aspect(aspect)
        self.title = self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.counter = self.ax.text(0.05, 1.15, "t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.iteration, system.engine.iterations), transform=self.ax.transAxes)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.initialized = True
        
    def update(self, system):
        """Update the view"""
        if not self.initialized:
            self.initialize(system)
        aspect = 1.0 / system.aspect * (system.nn / system.nz)
        self.ax.cla()
        self.ax.set_xlim(0, system.nx)
        self.ax.set_ylim(0, system.nz)
        self.ax.set_aspect(aspect)
        data = self.data(system)
        ptp = np.ptp(data)
        if np.isfinite(ptp) and ptp != 0.0:
            z,x = np.mgrid[:system.nz,:system.nn]
            u,v = data
            speed = np.sqrt(u**2.0 + v**2.0)
            self.image = self.ax.streamplot(x,z,u,v, color=speed, **self.im_kwargs)
            if hasattr(self, 'cb'):
                self.cb.ax.clear()
                self.cb = self.ax.figure.colorbar(self.image.lines, ax=self.ax, cax=self.cb.ax)
            else:
                self.cb = self.ax.figure.colorbar(self.image.lines, ax=self.ax)
        self.title = self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.counter.set_text("t={0.value:5.0f}{0.unit:generic} {1:4d}/{2:4d}".format(system.time, system.iteration, system.engine.iterations))
        

class ProfileView(View):
    """A profile of a single variable."""
    def __init__(self, variable):
        super(ProfileView, self).__init__()
        self.variable = variable
        self.line = None
        
    @abc.abstractmethod
    def ydata(self, system):
        """Get the ydata for this system"""
        pass
        
    @abc.abstractmethod
    def xdata(self, system):
        """Return the xdata."""
        pass
        
    def initialize(self, system):
        """Set up the plot."""
        super(ProfileView, self).initialize(system)
        self.line, = self.ax.plot(self.xdata(system), self.ydata(system), 'k-')
        self.ax.set_ylabel(getattr(type(system), self.variable).latex)
        
    def update(self, system):
        """Update plot."""
        super(ProfileView, self).update(system)
        self.line.set_data(self.xdata(system), self.ydata(system))
        self.ax.relim()
        self.ax.autoscale_view()
        
    
class MProfileView(ProfileView):
    """Modal Profile View"""
    def __init__(self, variable, z=1):
        super(MProfileView, self).__init__(variable)
        self.z = z
        
    def ydata(self, system):
        """Return the y-data values."""
        return getattr(system,self.variable).raw[self.z,:]
        
    def xdata(self, system):
        """Return the xdata."""
        return np.arange(system.nn)
        
    def initialize(self, system):
        """Initialize the plot view."""
        super(MProfileView, self).initialize(system)
        self.ax.set_title("[z={}] {} ({})".format(self.z, getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.ax.set_xlabel("Horizontal Mode")

class VProfileView(ProfileView):
    """docstring for VProfileView"""
    def __init__(self, variable, mode):
        super(VProfileView, self).__init__(variable)
        self.mode = mode
        
    def ydata(self, system):
        """Return the y-data values."""
        return getattr(system,self.variable).raw[:,self.mode]
        
    def xdata(self, system):
        """Return the xdata."""
        return np.arange(system.nz)
        
    def initialize(self, system):
        """Initialize the view."""
        super(VProfileView, self).initialize(system)
        self.ax.set_title("[k={}] {} ({})".format(self.mode, getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        self.ax.set_xlabel("z")

class V1DProfileView(VProfileView):
    """1st Derivative Profile vertically."""
    
    def ydata(self, system):
        """Return the y-data values."""
        from .finitedifference import first_derivative2D
        f = getattr(system,self.variable).raw
        df = np.zeros_like(f)
        fm = np.zeros(system.nn)
        fp = np.zeros(system.nn)
        if self.variable == "Temperature":
            fm[0] = 1.0
        assert not first_derivative2D(system.nz, system.nn, df, f, 1.0/system.nz, fp, fm, 1.0)
        return df[:,self.mode]
    
    def initialize(self, system):
        super(V1DProfileView, self).initialize(system)
        self.ax.set_ylabel(r"$\frac{{d{}}}{{d z}}$".format(getattr(type(system), self.variable).latex[1:-1]))


class V2DProfileView(VProfileView):
    """2nd Derivative Profile vertically."""
    
    def ydata(self, system):
        """Return the y-data values."""
        from .finitedifference import second_derivative2D
        f = getattr(system,self.variable).raw
        ddf = np.zeros_like(f)
        fm = np.zeros(system.nn)
        fp = np.zeros(system.nn)
        if self.variable == "Temperature":
            fm[0] = 1.0
        assert not second_derivative2D(system.nz, system.nn, ddf, f, 1.0/system.nz, fp, fm, 1.0)
        return ddf[:,self.mode]
    
    def initialize(self, system):
        super(V2DProfileView, self).initialize(system)
        self.ax.set_ylabel(r"$\frac{{d^2{}}}{{d z^2}}$".format(getattr(type(system), self.variable).latex[1:-1]))

class EvolutionView(View):
    """An object view showing the time-evolution of a parameter."""
    def __init__(self, variable):
        super(EvolutionView, self).__init__()
        self.variable = variable
        self.line = None
    
    def ydata(self, system):
        """Return the y-data values."""
        return system.engine[self.variable][...,:system.it]
        
    def xdata(self, system):
        """Return the x-data values."""
        return system.dimensional_full_array("Time")[:system.it]
    
    def initialize(self, system):
        """Set up the plot."""
        super(EvolutionView, self).initialize(system)
        self.line, = self.ax.plot(self.xdata(system), self.ydata(system), 'k-')
        self.ax.set_ylabel(getattr(type(system), self.variable).latex)
        self.ax.set_xlabel(type(system).Time.latex)
        self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
        
        
    def update(self, system):
        """Update the view."""
        super(EvolutionView, self).update(system)
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
        super(EvolutionViewAllModes, self).initialize(system)
        self.line = []
        for i in range(self.ydata(system).shape[1]):
            self.line.append(self.ax.plot(self.xdata(system), self.ydata(system)[self.zmode,i,:], 'k-')[0])
        self.ax.set_ylabel(getattr(type(system), self.variable).latex)
        self.ax.set_xlabel(type(system).Time.latex)
        self.ax.set_title("{} ({})".format(getattr(type(system), self.variable).name, getattr(type(system), self.variable).latex))
    
    def update(self, system):
        """Update the view."""
        super(EvolutionViewAllModes, self).update(system)
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
        

        
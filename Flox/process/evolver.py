# -*- coding: utf-8 -*-
# 
#  evolver.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-10.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

"""
Evolver managed by a process.
"""

from __future__ import (absolute_import, unicode_literals, division, print_function)

import multiprocessing.managers as mm
import multiprocessing as mp
import queue

from astropy.utils.console import ProgressBar

from .manager import AsynchronousManager, AsynchronousProxy

class EvolverManager(AsynchronousManager):
    """Manage Evolver objects"""
    
    def register_evolver(self, evolver):
        """Register an evolver with a generic proxy."""
        self.register(evolver.__name__, evolver.from_system, AsynchronousProxy)
    

class EvolverProcessing(object):
    """An object which manages processing."""
    def __init__(self, timeout=10, buffer_length=10):
        super(EvolverProcessing, self).__init__()
        self.async_manager = EvolverManager()
        self.queue_manager = mm.SyncManager()
        self.timeout = timeout
        self.buffer_length = buffer_length
    
    def register_evolver(self, evolver):
        """Register a specific type of evolver."""
        self.async_manager.register_evolver(evolver)
        
    def start(self):
        """Start the subprocesses."""
        self.async_manager.start()
        self.queue_manager.start()
        
    def stop(self):
        """Stop the subprocesses."""
        self.async_manager.stop()
        self.queue_manager.shutdown()
        self.async_manager.join()
        
    def __enter__(self):
        """Start this processor as a context."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit this processor as a context."""
        self.stop()
        
    def evolve(self, Evolver, System, time, chunks=int(1e3), chunksize=int(1e3), progress=True):
        """Evolve a given system."""
        Q = self.evolve_queue(Evolver, System, time, chunks, chunksize)
        for p in ProgressBar(chunks):
            System.read_packet(Q.get(timeout=self.timeout))
            
    def evolve_queue(self, Evolver, System, time, chunks=int(1e3), chunksize=int(1e3)):
        """Setup the evolver and return the queue."""
        E = getattr(self.async_manager, Evolver.__name__)(System)
        E.read_packet(System.create_packet())
        Q = self.queue_manager.Queue()
        nd_time = System.nondimensionalize(time).value
        E.evolve_async(nd_time, chunks=chunks, chunksize=chunksize, queue=Q)
        return Q
        
    def animate_evolve(self, Evolver, System, Plotter, time, chunks=int(1e3), chunksize=int(1e3), progress=True):
        """Animate an evolver"""
        from matplotlib import animation
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        Q = self.evolve_queue(Evolver, System, time, chunks, chunksize)
        with ProgressBar(chunks) as PBar:
            anim = animation.FuncAnimation(Plotter.figure, self._animate_callback, self._packet_callback(System, Q), interval=1, fargs=(System, Plotter, PBar))
            plt.show()
        
    
    
    def _packet_callback(self, System, Queue):
        """Packet callback method."""
        def _packet_generator():
            packet = None
            try:
                for j in range(self.buffer_length):
                    packet = Queue.get_nowait()
                    System.read_packet(packet)
            except queue.Empty:
                pass
            if packet is None:
                try:
                    packet = Queue.get(timeout=self.timeout)
                    System.read_packet(packet)
                except queue.Empty:
                    print("Ending Animation!")
                    raise StopIteration
            yield System.it
        
        return _packet_generator
            
    def _animate_callback(self, i, System, Plotter, PBar=None):
        """Animation Callback."""        
        if PBar is not None:
            PBar.update(System.it)
        if System.it > 2:
            Plotter.update(System)
        
        
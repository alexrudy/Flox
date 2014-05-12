# -*- coding: utf-8 -*-
# 
#  manager.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-10.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import itertools
import multiprocessing.managers as mm
import multiprocessing as mp

class Manager(mm.SyncManager):
    """A management class for Flox objects."""
    
    @classmethod
    def register_evolver(cls, klass):
        """Register a specific type of evolver for use with this class."""
        cls.register(klass.__name__, klass.from_system)
        
    
class AsynchronousManager(mp.Process):
    """An asynchronous manager, wrapped in a process."""
    def __init__(self, *args, **kwargs):
        super(AsynchronousManager, self).__init__(*args, **kwargs)
        self._input = mp.Queue()
        self._output = mp.Queue()
        self._synchronize = mp.Barrier(2)
        self._proxies = {}
        self._typecodes = {}
        self._object_ids = itertools.count()
    
    def run(self):
        """Run the asynchronous process."""
        worker = AsynchronousWorker(self._typecodes, self._input, self._output)
        worker.work()
        
    def stop(self):
        """Stop the asynchronous process."""
        self._input.put(("#ASYNC", "#STOP"))
        
    def __getattr__(self, attribute):
        """Passthrough to registered typecodes."""
        if attribute not in self._typecodes:
            raise AttributeError
        
        def initializer(*args, **kwargs):
            """Return a proxy object initialezed."""
            this_id = next(self._object_ids)
            self._input.put(("#ASYNC", "#INIT", attribute, this_id, args, kwargs))
            return self._proxies[attribute](this_id, self._input)
            
        return initializer
        
    def register(self, typecode, init_func, proxy_type):
        """Register a typecode."""
        self._typecodes[typecode] = init_func
        self._proxies[typecode] = proxy_type
        
class AsynchronousProxy(object):
    """A proxy object for Asynchronous worker objects."""
    def __init__(self, referent_id, input_queue):
        super(AsynchronousProxy, self).__init__()
        self.id = referent_id
        self.job = itertools.count()
        self._results = {}
        self._input = input_queue
        self.async = True
        
    def _call_async(self, method, args, kwargs):
        """Make an asynchronous call to the worker."""
        self._input.put(("#ASYNC", "#METHOD", self.id, method, args, kwargs))
        
    def _call_sync(self, method, args, kwargs):
        """Make a synchronous call to the worker."""
        self._input.put(("#SYNC", "#METHOD", self.id, method, args, kwargs))
        
    def __getattr__(self, method):
        """Attribute access """
        if self.async:
            _call = self._call_async
        else:
            _call = self._call_sync
        
        def caller(*args, **kwargs):
            """Method caller."""
            _call(method, args, kwargs)
            
        return caller

class AsynchronousWorker(object):
    """A worker for an asynchronous object."""
    def __init__(self, typecodes, input_queue, output_queue, timeout=None):
        super(AsynchronousWorker, self).__init__()
        self.typecodes = typecodes
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.timeout = timeout
        self._working = False
        self._objects = {}
        
    def handle_message(self, message):
        """Handle an incoming message."""
        sync, kind = message[:2]
        if kind == "#STOP":
            self._working = False
        elif kind == "#INIT":
            typecode, this_id, args, kwargs = message[2:]
            referent = self.typecodes[typecode](*args, **kwargs)
            self._objects[this_id] = referent
        elif kind == "#METHOD":
            this_id, method, args, kwargs = message[2:]
            result = getattr(self._objects[this_id], method)(*args, **kwargs)
            
    def work(self):
        """Run the working loop."""
        self._working = True
        while self._working:
            self.handle_message(self.input_queue.get(timeout=self.timeout))
        
class AsynchronousResult(object):
    """An asynchronous result"""
    def __init__(self, id, job):
        super(AsynchronousResult, self).__init__()
        self.job = job
        
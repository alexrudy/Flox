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

logger = mp.get_logger()

class RemoteError(Exception):
    """docstring for RemoteError"""
    def __init__(self, msg):
        super(RemoteError, self).__init__()
        self.msg = "REMOTE ERROR\n" + "-" * 72 + "\n" + msg
        

class AsynchronousProxy(object):
    """A proxy object for Asynchronous worker objects."""
    def __init__(self, referent_id, input_queue, output_sync, output_async):
        super(AsynchronousProxy, self).__init__()
        self.id = referent_id
        self.job = itertools.count()
        self._results = {}
        self._input = input_queue
        self._output_sync = output_sync
        self._output_async = output_async
        self.async = True
        
    def _call_async(self, method, args, kwargs):
        """Make an asynchronous call to the worker."""
        message = ("#ASYNC", "#METHOD", self.id, method, args, kwargs)
        self._input.put(message)
        logger.subdebug("Message Sent: {}".format(message))
        
    def _call_sync(self, method, args, kwargs):
        """Make a synchronous call to the worker."""
        message = ("#SYNC", "#METHOD", self.id, method, args, kwargs)
        self._input.put(message)
        logger.subdebug("Message Sent: {}".format(message))
        return self._handle_message(self._output_sync.get())
        
    def _handle_message(self, message):
        """Message handling."""
        logger.subdebug("Message Recieved: {}".format(message))
        sync, kind, result = message
        if kind == "#ERROR":
            raise RemoteError(result)
        if kind == "#TRACEBACK":
            raise result
        elif kind == "#RETURN":
            return result
        
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


class AsynchronousManager(mp.Process):
    """An asynchronous manager, wrapped in a process."""
    
    _proxies = {}
    _typecodes = {}
    
    def __init__(self, *args, **kwargs):
        super(AsynchronousManager, self).__init__(*args, **kwargs)
        self._input = mp.Queue()
        self._output_sync = mp.Queue()
        self._output_async = mp.Queue()
        self._object_ids = itertools.count()
    
    def run(self):
        """Run the asynchronous process."""
        worker = AsynchronousWorker(self._typecodes, self._input, self._output_sync, self._output_async)
        worker.work()
        
    def stop(self):
        """Stop the asynchronous process."""
        message = ("#ASYNC", "#STOP")
        self._input.put(message)
        logger.subdebug("Message Sent: {}".format(message))
        
    def __enter__(self):
        """Enter the context."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()
        
    def __getattr__(self, attribute):
        """Passthrough to registered typecodes."""
        if attribute not in self._typecodes:
            raise AttributeError("Unkown attribute '{}'".format(attribute))
        
        def initializer(*args, **kwargs):
            """Return a proxy object initialezed."""
            this_id = next(self._object_ids)
            message = ("#SYNC", "#INIT", attribute, this_id, args, kwargs)
            self._input.put(message)
            logger.subdebug("Message Sent: {}".format(message))
            reply_id = self._handle_message(self._output_sync.get(), "#ID")
            assert this_id == reply_id
            return self._proxies[attribute](this_id, self._input, self._output_sync, self._output_async)
            
        return initializer
        
    def send(self, value):
        """Pickle and send a single value, returning a proxy object for this type."""
        this_id = next(self._object_ids)
        message = ("#ASYNC", "#SENDVALUE", this_id, value)
        self._input.put(message)
        logger.subdebug("Message Sent: {}".format(message))
        return self._proxies.get(value.__class__.__name__, AsynchronousProxy)(this_id, self._input, self._output_sync, self._output_async)
        
    def _handle_message(self, message, code="#RETURN"):
        """Message handling."""
        logger.subdebug("Message Recieved: {}".format(message))
        sync, kind, result = message
        if kind == "#ERROR":
            raise RemoteError(result)
        elif kind == code:
            return result
        else:
            raise RemoteError(result)
        
    @classmethod
    def register(cls, typecode, init_func, proxy_type=AsynchronousProxy):
        """Register a typecode."""
        cls._typecodes[typecode] = init_func
        cls._proxies[typecode] = proxy_type

class AsynchronousWorker(object):
    """A worker for an asynchronous object."""
    def __init__(self, typecodes, input_queue, output_queue_sync, output_queue_async, timeout=None):
        super(AsynchronousWorker, self).__init__()
        self.typecodes = typecodes
        self.input_queue = input_queue
        self.output_queue_sync = output_queue_sync
        self.output_queue_async = output_queue_async
        self.timeout = timeout
        self._working = False
        self._objects = {}
        
    def reply(self, message):
        """Handle replies"""
        logger.subdebug("Message Sent: {}".format(message))
        sync = message[0]
        if sync == "#SYNC":
            self.output_queue_sync.put(message)
        else:
            if message[1] == "#ERROR":
                raise RemoteError(message[2])
            self.output_queue_async.put(message)
            
        
    def handle_message(self, message):
        """Handle an incoming message."""
        logger.subdebug("Message Recieved: {}".format(message))
        sync, kind = message[:2]
        if kind == "#STOP":
            self._working = False
        elif kind == "#INIT":
            typecode, this_id, args, kwargs = message[2:]
            try:
                referent = self.typecodes[typecode](*args, **kwargs)
                self._objects[this_id] = referent
            except Exception as e:
                message = (sync, "#TRACEBACK", e)
            else:
                message = (sync, "#ID", this_id)
            self.reply(message)
        elif kind == "#SENDVALUE":
            this_id, value = message[2:]
            self._objects[this_id] = value
            self.reply((sync, "#ID", this_id))
        elif kind == "#METHOD":
            this_id, method, args, kwargs = message[2:]
            try:
                method = getattr(self._objects[this_id], method)
            except AttributeError as e:
                message = (sync, "#ERROR", str(e))
            else:
                try:
                    result = method(*args, **kwargs)
                except Exception as e:
                    message = (sync, "#TRACEBACK", e)
                else:
                    message = (sync, "#RETURN", result)
            self.reply(message)
            
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
        
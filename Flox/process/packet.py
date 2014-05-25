# -*- coding: utf-8 -*-
# 
#  packet.py
#  flox
#  
#  Created by Alexander Rudy on 2014-05-09.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import abc
import six
from queue import Empty

@six.add_metaclass(abc.ABCMeta)
class PickleInterface(object):
    """The simple interface for declaring mutable and immutable parts."""
    
    def __getstate__(self):
        """Return the pickling state for this object. 
        Should return an empty dictionary, as we don't want to actually pickle the contents of this array engine."""
        return { name:getattr(self, name) for name in self.get_parameter_list() }
    
    def __setstate__(self, state):
        """Return the state, resetting the caching views."""
        [ self.__setattr__(name, value) for name, value in state.items() ]
        
    @abc.abstractclassmethod
    def get_parameter_list(cls):
        """This method should return the parameters which need to be pickled."""
        return []
        

class PacketInterface(PickleInterface):
    """The interface for packet consumers and producers."""
    
    @abc.abstractmethod
    def get_data_list(self):
        """This method should return the data parameters which can be sent via the Packet interface."""
        return []
    
    def check_array(self, value, name):
        """Check an array's value."""
        pass
        
    def create_packet(self):
        """Create a packet from the LinearEvolver state."""
        packet = dict()
        for variable in self.get_data_list():
            packet[variable] = getattr(self, variable)
        return packet
        
    def read_packet(self, packet):
        """Read an imcoming packet"""
        for variable in packet.keys():
            self.check_array(packet[variable], variable)
            setattr(self, variable, packet[variable])
            
    def read_queue(self, queue, timeout=None):
        """Read the packets off of a queue, consuming that queue."""
        packets = 0
        for i, q in enumerate(self.iterate_queue(queue, timeout=timeout)):
            packets = i
        return packets
        
    def iterate_queue(self, queue, timeout):
        """Returns an iterable over a queue."""
        while True:
            try:
                self.read_packet(queue.get(timeout=timeout))
            except Empty as e:
                raise StopIteration("Queue is Empty")
            else:
                yield self
        
    def iterate_queue_buffered(self, queue, timeout, buffer=10):
        """Iterate over a queue, but buffer the output results if many are available."""
        while True:
            try:
                for b in range(buffer-1):
                    self.read_packet(queue.get_nowait())
            except Empty as e:
                # Now the buffer is empty, but do nothing.
                pass
            # Wait for the last element to arrive.
            try:
                self.read_packet(queue.get(timeout=timeout))
            except Empty as e:
                raise StopIteration("Queue is Empty")
            else:
                yield self
        
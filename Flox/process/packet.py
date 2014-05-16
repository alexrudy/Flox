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

class Packet(dict):
    """A 2D packet, containing the data required for the setup of an evolver."""
    
    def __init__(self):
        super(Packet, self).__init__()
        
    def __setitem__(self, key, value):
        """Set the item, converting it to a numpy array in the process."""
        super(Packet, self).__setitem__(key, np.asanyarray(value))
        
@six.add_metaclass(abc.ABCMeta)
class PacketInterface(object):
    """The interface for packet consumers and producers."""
    
    def check_array(self, value, name):
        """Check an array's value."""
        pass
    
    def __iter__(self):
        """Return the iterator for this object."""
        raise NotImplementedError("Packet interface is not required to support iteration.")
    
    def __next__(self):
        """Support packet iteration."""
        raise NotImplementedError("Packet interface is not required to support iteration.")
        
    def __len__(self):
        """Support packet iteration."""
        raise NotImplementedError("Packet interface is not required to support iteration.")
    
    @abc.abstractmethod
    def get_packet_list(self):
        """Return the parameter list."""
        return []
    
    def create_packet(self):
        """Create a packet from the LinearEvolver state."""
        packet = Packet()
        for variable in self.get_packet_list():
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
        
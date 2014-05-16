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
        try:
            while True:
                packets += 1
                self.read_packet(queue.get(timeout=timeout))
        except Empty as e:
            pass
        return packets

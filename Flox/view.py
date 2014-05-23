# -*- coding: utf-8 -*-
# 
#  view.py
#  flox
#  
#  Created by Alexander Rudy on 2014-05-22.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import collections
import abc

class DictionaryView(collections.MutableMapping):
    """A dictionary view passthrough."""
    def __init__(self, source):
        super(DictionaryView, self).__init__()
        self.source = source
        
    def __getitem__(self, key):
        return self.source.__getitem__(key)
        
    def __setitem__(self, key, value):
        return self.source.__setitem__(key)
        
    def __delitem__(self, key):
        return self.source.__delitem__(key)
    
    def __iter__(self):
        return self.source.__iter__()
        
    def __len__(self):
        return self.source.__len__()
    
    
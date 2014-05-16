# -*- coding: utf-8 -*-
# 
#  util.py
#  EART275
#  
#  Created by Alexander Rudy on 2014-05-15.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import functools

def fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__
    
def callback_progressbar_wrapper(func, progressbar):
    """Construct a callback which works with the progressbar."""
    
    @functools.wraps(func)
    def _callback(iteration, *args):
        func(iteration, *args)
        progressbar.update(iteration)
    
    return _callback
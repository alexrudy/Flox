#!/usr/bin/env /opt/local/Library/Frameworks/Python.framework/Versions/3.3/bin/python3.3
# -*- coding: utf-8 -*-

import os.path, os
if "VIRTUAL_ENV" in os.environ:
    activate_this = os.path.join(os.environ["VIRTUAL_ENV"],'bin/activate_this.py')
    exec(open(activate_this).read(), dict(__file__=activate_this))

from Flox.manager import FloxManager

FloxManager().run()
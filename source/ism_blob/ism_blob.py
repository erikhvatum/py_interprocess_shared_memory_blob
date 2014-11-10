# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum, Zach Pincus

import sys
import contextlib

if sys.platform == 'win32':
    from . import ism_win32 as ism_impl
else:
    from . import ism_posix as ism_impl

ISMBlob = ism_impl.ISMBlob

@contextlib.contextmanager
def new_ism_array(name, shape, dtype, order='C', permissions=0o600):
    """Context manager for creating a shared interprocess numpy array:
    with new_ism_array('foo', (10,10), int) as arr:
        arr.fill(353)
        send_arr_to_other_process()
    
    After the with-block ends, the shared memory blob will be deleted unless
    another process has opened the array too.
    
    Note: accessing the array outside of the with block WILL segfault."""
    ism = ISMBlob.new(name, shape, dtype, order, permissions)
    try:
        yield ism.asarray()
    finally:
        ism.close()

@contextlib.contextmanager
def open_ism_array(name):
    """Context manager for opening an existing shared interprocess numpy array:
    with open_ism_array('foo') as arr:
        sum = arr.sum()
    
    After the with-block ends, the shared memory blob will be deleted unless
    other processes have the array still open.
    
    Note: accessing the array outside of the with block WILL segfault."""
    ism = ISMBlob.open(name)
    try:
        yield ism.asarray()
    finally:
        ism.close()


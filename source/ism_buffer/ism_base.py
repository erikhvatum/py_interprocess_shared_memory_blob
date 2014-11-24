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


import numpy
import json

class ISMBase:
    _MAGIC_COOKIE = b'\xF0\x0A'

    @classmethod
    def open(cls, name):
        """Open an existing shared interprocess numpy array:
        buffer = ISMBuffer.open('foo')
        arr = buffer.asarray()
        arr.fill(353)
        send_arr_to_other_process(arr)
        """
        
        return cls(name)

    @classmethod
    def new(cls, name, shape, dtype, order='C', permissions=0o600):
        """Create a shared interprocess numpy array:
        buffer = ISMBuffer.new('foo', (10,10), int)
        arr = buffer.asarray()
        arr.fill(353)
        send_arr_to_other_process(arr)"""
        
        dtype = numpy.dtype(dtype)
        dtype_str = numpy.lib.format.dtype_to_descr(dtype)
        size = numpy.multiply.reduce(shape, dtype=int) * dtype.itemsize
        descr = json.dumps((dtype_str, shape, order)).encode('ascii')
        return cls(name, create=True, permissions=permissions, size=size, descr=descr)

    def __init__(self, name, create, permissions, size, descr):
        pass
    
    def asarray(self):
        if self.closed:
            raise RuntimeError('operation on closed ISMBuffer')
        array = numpy.array(self, dtype=numpy.uint8, copy=False)
        if self.descr:
            dtype, shape, order = json.loads(self.descr.decode('ascii'))
            array = numpy.ndarray(shape, dtype=dtype, order=order, buffer=array)
        return array
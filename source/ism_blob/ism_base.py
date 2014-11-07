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
import pickle

class ISMBase:
    _MAGIC_COOKIE = b'\xF0\x0A'

    @classmethod
    def open_ism_array(cls, name):
        return cls(name).typed_numpy_view()

    @classmethod
    def new_ism_array(cls, name, dtype, shape, order='C', permissions=0o600):
        dtype = numpy.dtype(dtype)
        size = numpy.multiply.reduce(shape, dtype=int) * dtype.itemsize
        descr = pickle.dumps((dtype.descr, shape, order), protocol=-1)
        return cls(name, create=True, permissions=permissions, size=size, descr=descr).typed_numpy_view()

    def __init__(self, name, create=False, permissions=0o600, size=0, descr=b''):
        '''Note: The default value for createPermissions, 0o600, or 384, represents the unix permission "readable/writeable
        by owner".  This parameter is ignored on win32.'''
        raise NotImplementedError()

    def typed_numpy_view(self):
        array = numpy.array(self, dtype=numpy.uint8, copy=False)
        if self.descr:
            descr_type, shape, order = pickle.loads(self.descr)
            dtype = numpy.dtype(descr_type)
            array = numpy.ndarray(shape, dtype=dtype, order=order, buffer=array)
        return array
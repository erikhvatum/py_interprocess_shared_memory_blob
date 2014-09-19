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
# Authors: Erik Hvatum
#
# NB: OS X did not support sharing mutexes across processes until 10.8, at which point support was added for
# pthread_rwlock sharing only (at last check, the OS X pthread_rwlockattr_setpshared was out of date, erroneously
# stating that sharing is not supported).  pthread_mutexes are simpler, having only one flag, and about 2x faster
# accroding to some simple benchmarking.  In order to accomodate OS X, pthread_rwlocks are used instead, and
# only the pthread_rwlock write flag is utilized.

import ctypes
import errno
import numpy
import sys

c_uint16_p = ctypes.POINTER(ctypes.c_uint16)
c_uint32_p = ctypes.POINTER(ctypes.c_uint32)
MAP_FAILED = ctypes.cast(-1, ctypes.c_void_p)
PTHREAD_PROCESS_SHARED = 1

if sys.platform == 'linux':
    libc = ctypes.CDLL('libc.so.6')
    librt = ctypes.CDLL('librt.so.1')
    shm_open   = librt.shm_open
    shm_unlink = librt.shm_unlink
    ftruncate  = libc.ftruncate
    mmap       = libc.mmap
    munmap     = libc.munmap
    close      = libc.close
    pthread_rwlock_destroy = librt.pthread_rwlock_destroy
    pthread_rwlock_init = librt.pthread_rwlock_init
    pthread_rwlock_unlock = librt.pthread_rwlock_unlock
    pthread_rwlock_wrlock = librt.pthread_rwlock_wrlock
    pthread_rwlockattr_init = librt.pthread_rwlockattr_init
    pthread_rwlockattr_setpshared = librt.pthread_rwlockattr_setpshared
    O_RDONLY = 0
    O_RDWR   = 2
    O_CREAT  = 64
    O_EXCL   = 128
    O_TRUNC  = 512
    PROT_READ  = 1
    PROT_WRITE = 2
    MAP_SHARED = 1
    # NB: 3rd argument, mode_t, is 4 bytes on linux and 2 bytes on osx (64 bit linux and osx, that is.  32
    # bit?  Refer to: http://dilbert.com/strips/comic/1995-06-24/ ...)
    shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
    c_rwlockattr_t = ctypes.c_byte * 8
elif sys.platform == 'darwin':
    libc = ctypes.CDLL('libc.dylib')
    shm_open   = libc.shm_open
    shm_unlink = libc.shm_unlink
    ftruncate  = libc.ftruncate
    mmap       = libc.mmap
    munmap     = libc.munmap
    close      = libc.close
    pthread_rwlock_destroy = libc.pthread_rwlock_destroy
    pthread_rwlock_init = libc.pthread_rwlock_init
    pthread_rwlock_unlock = libc.pthread_rwlock_unlock
    pthread_rwlock_wrlock = libc.pthread_rwlock_wrlock
    pthread_rwlockattr_init = libc.pthread_rwlockattr_init
    pthread_rwlockattr_setpshared = libc.pthread_rwlockattr_setpshared
    O_RDONLY = 0
    O_RDWR   = 2
    O_CREAT  = 512
    O_EXCL   = 2048
    O_TRUNC  = 1024
    PROT_READ  = 1
    PROT_WRITE = 2
    MAP_SHARED = 1
    shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint16]
    c_rwlockattr_t = ctypes.c_byte * 24
else:
    raise NotImplementedError('Platform "{}" is unsupported.  Only linux and darwin are supported.')

c_rwlockattr_t_p = ctypes.POINTER(c_rwlockattr_t)
shm_unlink.argtypes = [ctypes.c_char_p]
ftruncate.argtypes = [ctypes.c_int, ctypes.c_int64]
mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
mmap.restype = ctypes.c_void_p
munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
close.argtypes = [ctypes.c_int]

class ISMBlob:
    def __init__(self, name, shape, create=True, own=True):
        self.own = own
        self.fd = None
        self.data = None
        self.name = name.encode('utf-8') if type(name) is str else name
        self.byteCount = numpy.array(shape).prod() * 2
        self.ndarray = None
        if create:
            # NB: 0o600 represents the unix permission value readable/writeable by owner
            self.fd = shm_open(self.name, O_RDWR | O_CREAT | O_EXCL, 0o600)
            if self.fd == -1:
                self._rose('shm_open')
            if ftruncate(self.fd, self.byteCount) == -1:
                self._rose('ftruncate')
        else:
            self.fd = shm_open(self.name, O_RDWR, 0)
            if self.fd == -1:
                self._rose('shm_open')
        data = mmap(ctypes.c_void_p(0), self.byteCount, PROT_READ | PROT_WRITE, MAP_SHARED, self.fd, 0)
        if data == MAP_FAILED or data == ctypes.c_void_p(0):
            self._rose('mmap')
        self.data = ctypes.cast(data, c_uint16_p)
        self.ndarray = numpy.ctypeslib.as_array(self.data, shape)

    def __del__(self):
        del self.ndarray
        if self.data is not None:
            munmap(ctypes.cast(self.data, ctypes.c_void_p), self.byteCount)
        if self.fd is not None:
            if close(self.fd) == -1:
                self._rose('close')
            if self.own:
                if shm_unlink(self.name) == -1:
                    self._rose('shm_unlink')

    @staticmethod
    def _rose(fname):
        e = ctypes.get_errno()
        if e == 0:
            raise RuntimeError(fname + ' failed, but errno is 0.')
        raise OSError(e, errno.errorcode.get(e, 'UNKNOWN ERROR'))

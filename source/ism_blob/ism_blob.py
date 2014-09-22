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
# pthread_rwlock sharing only (at last check, the OS X pthread_rwlockattr_setpshared man page was out of date,
# falsely stating that sharing is not supported).  pthread_mutexes are simpler, having only one flag, and are
# about 2x faster accroding to some simple benchmarking.  So, in order to accomodate OS X, pthread_rwlocks are
# used instead, and only the pthread_rwlock write flag is utilized.

import contextlib
import ctypes
import errno
import numpy
import sys
import threading

c_uint16_p = ctypes.POINTER(ctypes.c_uint16)
c_uint32_p = ctypes.POINTER(ctypes.c_uint32)
c_uint64_p = ctypes.POINTER(ctypes.c_uint64)
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
    pthread_rwlockattr_destroy = librt.pthread_rwlockattr_destroy
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
    c_pthread_rwlockattr_t = ctypes.c_byte * 8
    c_pthread_rwlock_t = ctypes.c_byte * 56
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
    pthread_rwlockattr_destroy = libc.pthread_rwlockattr_destroy
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
    c_pthread_rwlockattr_t = ctypes.c_byte * 24
    c_pthread_rwlock_t = ctypes.c_byte * 200
else:
    raise NotImplementedError('Platform "{}" is unsupported.  Only linux and darwin are supported.')

def _pthread_errcheck(funcName, result, func, args):
    if result != 0:
        raise OSError(result, '{}(..) failed: {}'.format(funcName, errno.errorcode.get(result, 'UNKNOWN ERROR')))
    return result

def _osfunc_errcheck(funcName, result, func, args):
    if result < 0:
        e = ctypes.get_errno()
        if e == 0:
            raise RuntimeError(funcName + ' failed, but errno is 0.')
        raise OSError(e, funcName + ' failed: ' + str(errno.errorcode.get(e, 'UNKNOWN ERROR')))
    return result

def _mmap_errcheck(result, func, args):
    if result == 0 or result == MAP_FAILED:
        e = ctypes.get_errno()
        if e == 0:
            raise RuntimeError('mmap failed, but errno is 0.')
        raise OSError(e, 'mmap failed: ' + errno.errorcode.get(e, 'UNKNOWN ERROR'))
    return result

c_pthread_rwlockattr_t_p = ctypes.POINTER(c_pthread_rwlockattr_t)
c_pthread_rwlock_t_p = ctypes.POINTER(c_pthread_rwlock_t)
shm_unlink.argtypes = [ctypes.c_char_p]
ftruncate.argtypes = [ctypes.c_int, ctypes.c_int64]
mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
mmap.restype = ctypes.c_void_p
mmap.errcheck = _mmap_errcheck
munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
close.argtypes = [ctypes.c_int]
pthread_rwlock_destroy.argtypes = [c_pthread_rwlock_t_p]
pthread_rwlock_init.argtypes = [c_pthread_rwlock_t_p, c_pthread_rwlockattr_t_p]
pthread_rwlock_unlock.argtypes = [c_pthread_rwlock_t_p]
pthread_rwlock_wrlock.argtypes = [c_pthread_rwlock_t_p]
pthread_rwlockattr_destroy.argtypes = [c_pthread_rwlockattr_t_p]
pthread_rwlockattr_init.argtypes = [c_pthread_rwlockattr_t_p]
pthread_rwlockattr_setpshared.argtypes = [c_pthread_rwlockattr_t_p, ctypes.c_int]
for funcName in ('pthread_rwlock_destroy',
                'pthread_rwlock_init',
                'pthread_rwlock_unlock',
                'pthread_rwlock_wrlock',
                'pthread_rwlockattr_destroy',
                'pthread_rwlockattr_init',
                'pthread_rwlockattr_setpshared'):
    eval(funcName).errcheck = lambda result, func, args, funcName=funcName: _pthread_errcheck(funcName, result, func, args)
for funcName in ('shm_open',
                 'shm_unlink',
                 'ftruncate',
                 'munmap',
                 'close'):
    eval(funcName).errcheck = lambda result, func, args, funcName=funcName: _osfunc_errcheck(funcName, result, func, args)
del funcName

class _ISMBlobHeader(ctypes.Structure):
    _fields_ = [('refCountLock', c_pthread_rwlock_t),
                ('refCount', ctypes.c_uint64),
                ('userSize', ctypes.c_uint64)]

_ISMBlobHeader_p = ctypes.POINTER(_ISMBlobHeader)

@contextlib.contextmanager
def _refCountLock(p_header):
    pthread_rwlock_wrlock(ctypes.byref(p_header[0].refCountLock))
    yield
    pthread_rwlock_unlock(ctypes.byref(p_header[0].refCountLock))

class ISMBlob:
    def __init__(self, name, size, create=False, createPermissions=0o600):
        '''Note: 0o600 represents the unix permission value readable/writeable by owner'''
        if size <= 0:
            raise ValueError('size must be a positive integer (or floating point value, which will be rounded down).')
        self._fd = None
        self._data = None
        self._p_header = None
        # Used to determine if destructor should destroy shared region upon allocation failure
        self._isCreator = create
        self._name = str(name).encode('utf-8') if type(name) is not bytes else name
        self._userSize = size
        self._size = size + ctypes.sizeof(_ISMBlobHeader)
        if create:
            self._fd = shm_open(self._name, O_RDWR | O_CREAT | O_EXCL, createPermissions)
            ftruncate(self._fd, self._size)
        else:
            self._fd = shm_open(self._name, O_RDWR, 0)
        self._data = mmap(ctypes.c_void_p(0), self._size, PROT_READ | PROT_WRITE, MAP_SHARED, self._fd, 0)
        self._userData = self._data + ctypes.sizeof(_ISMBlobHeader)
        self._p_header = ctypes.cast(self._data, _ISMBlobHeader_p)
        if create:
            lockAttr = c_pthread_rwlockattr_t()
            p_lockAttr = ctypes.pointer(lockAttr)
            pthread_rwlockattr_init(p_lockAttr)
            pthread_rwlockattr_setpshared(p_lockAttr, PTHREAD_PROCESS_SHARED)
            pthread_rwlock_init(ctypes.byref(self._p_header[0].refCountLock), p_lockAttr)
            pthread_rwlockattr_destroy(p_lockAttr)
            with _refCountLock(self._p_header):
                self._p_header[0].refCount = 1
                self._p_header[0].userSize = self._userSize
        else:
            with _refCountLock(self._p_header):
                self._p_header[0].refCount += 1
            if self._p_header[0].userSize != self._userSize:
                raise ValueError('Supplied size ({}) does not match shared region size ({}).'.format(self._userSize, self._p_header[0].userSize))

    def __del__(self):
        if self._p_header is not None:
            destroy = False
            try:
                # The refCount lock resides within the shared memory region to be destroyed and thus does not prevent
                # another thread in the same process from initiating a deep copy of this ISM object in the interval between
                # refCount lock release and completion of this destructor function.  However, if this destructor is executing,
                # it is as a consequence of no Python references to this object remaining, precluding the possibility of a copy
                # operation being initiated except behind the interpreter's back.  Therefore, a second in-process lock
                # to guard against this is not required.  Additionally, if refCount reaches zero below, no other processes
                # are still referring to the shared memory, precluding the possibility of a copy operation in *another process*
                # during the interval between refLock destruction and completion of this destructor function, so a second
                # shared-process lock to guard against this is not required.
                #
                # The purpose of locating this logic within try/finally is to ensure correct behavior even if ctrl-c is received
                # during destructor execution.
                with _refCountLock(self._p_header):
                    self._p_header[0].refCount -= 1
                    if self._p_header[0].refCount == 0:
                        destroy = True
            finally:
                if destroy:
                    munmap(self._data, self._size)
                    close(self._fd)
                    shm_unlink(self._name)
                    return
        elif self._fd is not None:
            close(self._fd)
            if self._isCreator:
                # Oops, allocation failed.  Get rid of the partially initialized shared memory region.
                shm_unlink(self._name)

    @property
    def name(self):
        return self._name.decode('utf-8')

    @property
    def size(self):
        return self._userSize

    @property
    def data(self):
        return self._userData        

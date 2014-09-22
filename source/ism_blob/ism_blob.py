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

# NB: OS X did not support sharing mutexes across processes until 10.8, at which point support was added for
# pthread_rwlock sharing only (at last check, the OS X pthread_rwlockattr_setpshared man page was out of date,
# falsely stating that sharing is not supported).  pthread_mutexes are simpler, having only one flag, and are
# about 2x faster accroding to some simple benchmarking.  So, in order to accomodate OS X, pthread_rwlocks are
# used instead, and only the pthread_rwlock write flag is utilized.

import contextlib
import ctypes
import errno
import functools
import numpy
import sys
import threading

_MAP_FAILED = ctypes.cast(-1, ctypes.c_void_p)
_PTHREAD_PROCESS_SHARED = 1

if sys.platform == 'linux':
    _libc = ctypes.CDLL('libc.so.6')
    _librt = ctypes.CDLL('librt.so.1')
    _shm_open   = _librt.shm_open
    _shm_unlink = _librt.shm_unlink
    _ftruncate  = _libc.ftruncate
    _mmap       = _libc.mmap
    _munmap     = _libc.munmap
    _close      = _libc.close
    _pthread_rwlock_destroy = _librt.pthread_rwlock_destroy
    _pthread_rwlock_init = _librt.pthread_rwlock_init
    _pthread_rwlock_unlock = _librt.pthread_rwlock_unlock
    _pthread_rwlock_wrlock = _librt.pthread_rwlock_wrlock
    _pthread_rwlockattr_destroy = _librt.pthread_rwlockattr_destroy
    _pthread_rwlockattr_init = _librt.pthread_rwlockattr_init
    _pthread_rwlockattr_setpshared = _librt.pthread_rwlockattr_setpshared
    _O_RDONLY = 0
    _O_RDWR   = 2
    _O_CREAT  = 64
    _O_EXCL   = 128
    _PROT_READ  = 1
    _PROT_WRITE = 2
    _MAP_SHARED = 1
    # NB: 3rd argument, mode_t, is 4 bytes on linux and 2 bytes on osx (64 bit linux and osx, that is.  32
    # bit?  Refer to: http://dilbert.com/strips/comic/1995-06-24/ ...)
    _shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
    _c_pthread_rwlockattr_t = ctypes.c_byte * 8
    _c_pthread_rwlock_t = ctypes.c_byte * 56
elif sys.platform == 'darwin':
    _libc = ctypes.CDLL('libc.dylib')
    _shm_open   = _libc.shm_open
    _shm_unlink = _libc.shm_unlink
    _ftruncate  = _libc.ftruncate
    _mmap       = _libc.mmap
    _munmap     = _libc.munmap
    _close      = _libc.close
    _pthread_rwlock_destroy = _libc.pthread_rwlock_destroy
    _pthread_rwlock_init = _libc.pthread_rwlock_init
    _pthread_rwlock_unlock = _libc.pthread_rwlock_unlock
    _pthread_rwlock_wrlock = _libc.pthread_rwlock_wrlock
    _pthread_rwlockattr_destroy = _libc.pthread_rwlockattr_destroy
    _pthread_rwlockattr_init = _libc.pthread_rwlockattr_init
    _pthread_rwlockattr_setpshared = _libc.pthread_rwlockattr_setpshared
    _O_RDONLY = 0
    _O_RDWR   = 2
    _O_CREAT  = 512
    _O_EXCL   = 2048
    _PROT_READ  = 1
    _PROT_WRITE = 2
    _MAP_SHARED = 1
    _shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint16]
    _c_pthread_rwlockattr_t = ctypes.c_byte * 24
    _c_pthread_rwlock_t = ctypes.c_byte * 200
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
    if result == 0 or result == _MAP_FAILED:
        e = ctypes.get_errno()
        if e == 0:
            raise RuntimeError('mmap failed, but errno is 0.')
        raise OSError(e, 'mmap failed: ' + errno.errorcode.get(e, 'UNKNOWN ERROR'))
    return result

_c_pthread_rwlockattr_t_p = ctypes.POINTER(_c_pthread_rwlockattr_t)
_c_pthread_rwlock_t_p = ctypes.POINTER(_c_pthread_rwlock_t)
_shm_unlink.argtypes = [ctypes.c_char_p]
_ftruncate.argtypes = [ctypes.c_int, ctypes.c_int64]
_mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
_mmap.restype = ctypes.c_void_p
_mmap.errcheck = _mmap_errcheck
_munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_close.argtypes = [ctypes.c_int]
_pthread_rwlock_destroy.argtypes = [_c_pthread_rwlock_t_p]
_pthread_rwlock_init.argtypes = [_c_pthread_rwlock_t_p, _c_pthread_rwlockattr_t_p]
_pthread_rwlock_unlock.argtypes = [_c_pthread_rwlock_t_p]
_pthread_rwlock_wrlock.argtypes = [_c_pthread_rwlock_t_p]
_pthread_rwlockattr_destroy.argtypes = [_c_pthread_rwlockattr_t_p]
_pthread_rwlockattr_init.argtypes = [_c_pthread_rwlockattr_t_p]
_pthread_rwlockattr_setpshared.argtypes = [_c_pthread_rwlockattr_t_p, ctypes.c_int]
for funcName in ('_pthread_rwlock_destroy',
                '_pthread_rwlock_init',
                '_pthread_rwlock_unlock',
                '_pthread_rwlock_wrlock',
                '_pthread_rwlockattr_destroy',
                '_pthread_rwlockattr_init',
                '_pthread_rwlockattr_setpshared'):
    eval(funcName).errcheck = lambda result, func, args, funcName=funcName: _pthread_errcheck(funcName, result, func, args)
for funcName in ('_shm_open',
                 '_shm_unlink',
                 '_ftruncate',
                 '_munmap',
                 '_close'):
    eval(funcName).errcheck = lambda result, func, args, funcName=funcName: _osfunc_errcheck(funcName, result, func, args)
del funcName

class _ISMBlobHeader(ctypes.Structure):
    _fields_ = [('refCountLock', _c_pthread_rwlock_t),
                ('refCount', ctypes.c_uint64),
                ('userSize', ctypes.c_uint64)]

_ISMBlobHeader_p = ctypes.POINTER(_ISMBlobHeader)

@contextlib.contextmanager
def _refCountLock(p_header):
    _pthread_rwlock_wrlock(ctypes.byref(p_header[0].refCountLock))
    yield
    _pthread_rwlock_unlock(ctypes.byref(p_header[0].refCountLock))

_dts_to_cts = {}
for t in (ctypes.c_float, ctypes.c_double):
    _dts_to_cts[">f%s" % ctypes.sizeof(t)] = t.__ctype_be__
    _dts_to_cts["<f%s" % ctypes.sizeof(t)] = t.__ctype_le__
for t in (ctypes.c_byte, ctypes.c_short, ctypes.c_int, ctypes.c_long, ctypes.c_longlong):
    _dts_to_cts[">i%s" % ctypes.sizeof(t)] = t.__ctype_be__
    _dts_to_cts["<i%s" % ctypes.sizeof(t)] = t.__ctype_le__
for t in (ctypes.c_ubyte, ctypes.c_ushort, ctypes.c_uint, ctypes.c_ulong, ctypes.c_ulonglong):
    _dts_to_cts[">u%s" % ctypes.sizeof(t)] = t.__ctype_be__
    _dts_to_cts["<u%s" % ctypes.sizeof(t)] = t.__ctype_le__
del t
_dts_to_cts["|b1"] = ctypes.c_bool
_dts_to_cts["|i1"] = ctypes.c_byte
_dts_to_cts["|u1"] = ctypes.c_ubyte

def _get_ctype(dtype):
    '''Get the ctype numpy uses internally to represent dtype.'''
    try:
        return _dts_to_cts[dtype.descr[0][1]]
    except KeyError:
        raise ValueError("Cannot convert dtype to ctype: {0}".format(dt))

class ISMBlob:
    '''ISMBlob, "Interprocess Shared Memory Blob", provides allocation, reference counting, and automatic deallocation
    of an interprocess shared memory region identified by a name string.  The ISMBlob.create_with_numpy_view and
    ISMBlob.open_with_numpy_view factory functions are available for the special case of creating a new numpy ndarray
    in shared memroy and for obtaining a reference to an existing shared ndarray, respectively.  For example, in Python
    process A:
        import numpy
        import ism_blob
        ablob, a = ism_blob.ISMBlob.create_with_numpy_view('foo', (10,10), numpy.uint32)
        a[4,5] = 120
    Subsequently, in Python process B:
        import numpy
        import ism_blob
        ablob, a = ism_blob.ISMBlob.open_with_numpy_view('foo', (10,10), numpy.uint32)
        print(a[4,5])
    .. 120

    A shared memory region is destroyed when the last ISMBlob referring to it is destroyed.  As a consequence, when using
    the create/open_with_numpy_view functions, the ISMBlob returned with the array should be retained while the array
    exists.'''
    def __init__(self, name, size, create=False, createPermissions=0o600):
        '''Note: 0o600, or 384, represents the unix permission "readable/writeable by owner".'''
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
            self._fd = _shm_open(self._name, _O_RDWR | _O_CREAT | _O_EXCL, createPermissions)
            _ftruncate(self._fd, self._size)
        else:
            self._fd = _shm_open(self._name, _O_RDWR, 0)
        self._data = _mmap(ctypes.c_void_p(0), self._size, _PROT_READ | _PROT_WRITE, _MAP_SHARED, self._fd, 0)
        self._userData = self._data + ctypes.sizeof(_ISMBlobHeader)
        self._p_header = ctypes.cast(self._data, _ISMBlobHeader_p)
        if create:
            lockAttr = _c_pthread_rwlockattr_t()
            p_lockAttr = ctypes.pointer(lockAttr)
            _pthread_rwlockattr_init(p_lockAttr)
            _pthread_rwlockattr_setpshared(p_lockAttr, _PTHREAD_PROCESS_SHARED)
            _pthread_rwlock_init(ctypes.byref(self._p_header[0].refCountLock), p_lockAttr)
            _pthread_rwlockattr_destroy(p_lockAttr)
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
                    _munmap(self._data, self._size)
                    _close(self._fd)
                    _shm_unlink(self._name)
                    return
        elif self._fd is not None:
            _close(self._fd)
            if self._isCreator:
                # Oops, allocation failed.  Get rid of the partially initialized shared memory region.
                _shm_unlink(self._name)

    @classmethod
    def create_with_numpy_view(cls, name, shape, dtype, permissions=0o600):
        '''Returns a tuple containing a new ISMBlob and a numpy ndarray that is a view of the ISMBlob's shared memory region.
        The ISMBlob created is identified by the value in the name argument and is of size sufficient to hold the requested
        ndarray.  Note that the memory backing the ndarray is reference counted by ISMBlob; when the last ISMBlob with the
        same name across every process on the system referring to that region is deleted, the ndarray's backing memory is
        freed.  It is advisable to retain a reference to the ISMBlob while its associated ndarray exists.

        Note: 0o600, or 384, represents the unix permission "readable/writeable by owner".'''
        dt = numpy.dtype(dtype)
        ct = _get_ctype(dt)
        pct = ctypes.POINTER(ct)
        size = ctypes.sizeof(ct) * functools.reduce(lambda a,b:a*b, shape)
        ismb = cls(name, size, True, permissions)
        ndarray = numpy.ctypeslib.as_array(ctypes.cast(ismb.data, pct), shape=shape)
        return ismb, ndarray

    @classmethod
    def open_with_numpy_view(cls, name, shape, dtype):
        '''Returns a tuple containing an ISMBlob referring to an existing shared memory blob.  Note that the memory backing the 
        ndarray is reference counted by ISMBlob; when the last ISMBlob with the same name across every process on the system 
        referring to that region is deleted, the ndarray's backing memory is freed.  It is advisable to retain a reference to 
        the ISMBlob while its associated ndarray exists.'''
        dt = numpy.dtype(dtype)
        ct = _get_ctype(dt)
        pct = ctypes.POINTER(ct)
        size = ctypes.sizeof(ct) * functools.reduce(lambda a,b:a*b, shape)
        ismb = cls(name, size)
        ndarray = numpy.ctypeslib.as_array(ctypes.cast(ismb.data, pct), shape=shape)
        return ismb, ndarray

    @property
    def name(self):
        return self._name.decode('utf-8')

    @property
    def size(self):
        return self._userSize

    @property
    def data(self):
        return self._userData

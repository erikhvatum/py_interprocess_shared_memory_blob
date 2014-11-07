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

# NB: OS X did not support sharing mutexes across processes until 10.8, at which point support was added for
# pthread_rwlock sharing only (at last check, the OS X pthread_rwlockattr_setpshared man page was out of date,
# falsely stating that sharing is not supported).  pthread_mutexes are simpler, having only one flag, and are
# about 2x faster accroding to some simple benchmarking.  However, this difference is dwarfed by Python
# interpreter overhead, and in order to avoid adding a third implementation, the POSIX implementation
# accomodates OS X by using pthread_rwlocks (only the pthread_rwlock write flag is utilized). win32 does its
# thing, as always.

import contextlib
import ctypes
import errno
import os
import sys

from . import ism_base

# Set up ctype types and wrappers for various system API functions.  The differences between the Linux and OS X
# calls and types that we require are slim, allowing both to share code after this section.

if sys.platform == 'linux':
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    librt = ctypes.CDLL('librt.so.1', use_errno=True)
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
    PROT_READ  = 1
    PROT_WRITE = 2
    MAP_SHARED = 1
    # NB: 3rd argument, mode_t, is 4 bytes on linux and 2 bytes on osx (64 bit linux and osx, that is.  I
    # haven't had a chance to try this on 32-bit platforms, but patches / pull requests are welcome.)
    shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
    pthread_rwlockattr_t = ctypes.c_byte * 8
    pthread_rwlock_t = ctypes.c_byte * 56
    
elif sys.platform == 'darwin':
    libc = ctypes.CDLL('libc.dylib', use_errno=True)
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
    PROT_READ  = 1
    PROT_WRITE = 2
    MAP_SHARED = 1
    shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint16]
    pthread_rwlockattr_t = ctypes.c_byte * 24
    pthread_rwlock_t = ctypes.c_byte * 200
    
else:
    raise NotImplementedError("ism_blob's POSIX implementation is currently only for Linux and Darwin")

MAP_FAILED = ctypes.c_void_p(-1).value
NULL_P = ctypes.c_void_p(0)
PTHREAD_PROCESS_SHARED = 1

def pthread_errcheck(funcName, result, func, args):
    if result != 0:
        raise OSError(result, '{}(..) failed: {}'.format(funcName, describe_sys_errno(result)))
    return result

def osfunc_errcheck(funcName, result, func, args):
    if result < 0:
        e = ctypes.get_errno()
        if e == 0:
            raise RuntimeError(funcName + ' failed, but errno is 0.')
        raise OSError(e, funcName + ' failed: ' + describe_sys_errno(e))
    return result

def mmap_errcheck(result, func, args):
    if result == 0 or result == MAP_FAILED:
        e = ctypes.get_errno()
        if e == 0:
            raise RuntimeError('mmap failed, but errno is 0.')
        raise OSError(e, 'mmap failed: ' + describe_sys_errno(e))
    return result

def describe_sys_errno(e):
    try:
        strerror = os.strerror(e)
    except ValueError:
        strerror = 'no description available'
    return '{} ({})'.format(strerror, errno.errorcode.get(e, 'UNKNOWN ERROR'))

pthread_rwlockattr_t_p = ctypes.POINTER(pthread_rwlockattr_t)
pthread_rwlock_t_p = ctypes.POINTER(pthread_rwlock_t)
shm_unlink.argtypes = [ctypes.c_char_p]
ftruncate.argtypes = [ctypes.c_int, ctypes.c_int64]
mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
mmap.restype = ctypes.c_void_p
mmap.errcheck = mmap_errcheck
munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
close.argtypes = [ctypes.c_int]
pthread_rwlock_destroy.argtypes = [pthread_rwlock_t_p]
pthread_rwlock_init.argtypes = [pthread_rwlock_t_p, pthread_rwlockattr_t_p]
pthread_rwlock_unlock.argtypes = [pthread_rwlock_t_p]
pthread_rwlock_wrlock.argtypes = [pthread_rwlock_t_p]
pthread_rwlockattr_destroy.argtypes = [pthread_rwlockattr_t_p]
pthread_rwlockattr_init.argtypes = [pthread_rwlockattr_t_p]
pthread_rwlockattr_setpshared.argtypes = [pthread_rwlockattr_t_p, ctypes.c_int]
for funcName in (
    'pthread_rwlock_destroy',
    'pthread_rwlock_init',
    'pthread_rwlock_unlock',
    'pthread_rwlock_wrlock',
    'pthread_rwlockattr_destroy',
    'pthread_rwlockattr_init',
    'pthread_rwlockattr_setpshared'):
    eval(funcName).errcheck = lambda result, func, args, funcName=funcName: pthread_errcheck(funcName, result, func, args)
for funcName in (
    'shm_open',
    'shm_unlink',
    'ftruncate',
    'munmap',
    'close'):
    eval(funcName).errcheck = lambda result, func, args, funcName=funcName: osfunc_errcheck(funcName, result, func, args)
del funcName


# layout of ISMBlob: SizeHeader, descr, data, RefCountHeader
# RefCountHeader is last so that clients that don't care about refcounting
# and make the server just hold onto the buffer can just ignore it.
class SizeHeader(ctypes.Structure):
    _fields_ = [
        ('magic_cookie', ctypes.c_char*len(ism_base.ISMBase._MAGIC_COOKIE)),
        ('descr_size', ctypes.c_uint16),
        ('data_size', ctypes.c_uint64)
    ]

class RefCountHeader(ctypes.Structure):
    _fields_ = [
        ('refcount_lock', pthread_rwlock_t),
        ('refcount', ctypes.c_uint64),
    ]

SizeHeaderSize = ctypes.sizeof(SizeHeader)

class ISMBlob(ism_base.ISMBase):    
    def __init__(self, name, create=False, permissions=0o600, size=0, descr=b''):
        '''Note: The default value for createPermissions, 0o600, or 384, represents the unix permission "readable/writeable
        by owner".'''
        self._all_allocated = False
        self._name = str(name).encode('utf-8') if type(name) is not bytes else name
        try:
            # first, figure out the sizes of everything. Easy if we're creating the blob; requires a bit of digging if not
            if create:
                self.size = size
                descr_size = len(descr)
            else:
                self._fd = shm_open(self._name, O_RDWR, 0)
                size_header_addr = mmap(NULL_P, SizeHeaderSize, PROT_READ, MAP_SHARED, self._fd, 0)
                try:
                    size_header = SizeHeader.from_address(size_header_addr)
                    assert size_header.magic_cookie == self._MAGIC_COOKIE
                    self.size = size_header.data_size
                    descr_size = size_header.descr_size
                finally:
                    munmap(size_header_addr, SizeHeaderSize)
        
            class DataLayout(ctypes.Structure):
                _fields_ = [
                    ('size_header', SizeHeader),
                    ('descr', ctypes.c_ubyte*descr_size),
                    ('data', ctypes.c_ubyte*self.size),
                    ('refcount_header', RefCountHeader)
                ]
            self._mmap_size = ctypes.sizeof(DataLayout)
        
            if create:
                # if we're creating it, open the fd now. If it was extant, the fd already got opened above
                self._fd = shm_open(self._name, O_RDWR | O_CREAT | O_EXCL, permissions)
                os.ftruncate(self._fd, self._mmap_size)
            
            self._mmap_data = mmap(NULL_P, self._mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, self._fd, 0)
            # TODO: do we need MAP_HASSEMAPHORE above as well??
            data_layout = DataLayout.from_address(self._mmap_data)
            self._refcount_header = data_layout.refcount_header
            self.data = data_layout.data
            self.__array_interface__ = {
                'shape': (self.size,),
                'typestr':'|u1',
                'version':3,
                'data':(ctypes.addressof(self.data), False)
            }
        
            if create:
                # fill in the header information and create the rwlock
                size_header = data_layout.size_header
                size_header.descr_size = descr_size
                size_header.data_size = self.size
                self.descr = descr
                ctypes.memmove(data_layout.descr, descr, descr_size)
                lockattr = pthread_rwlockattr_t()
                lockattr_ref = ctypes.byref(lockattr)
                pthread_rwlockattr_init(lockattr_ref)
                try:
                    pthread_rwlockattr_setpshared(lockattr_ref, PTHREAD_PROCESS_SHARED)
                    _refcount_lock = ctypes.byref(self._refcount_header.refcount_lock)
                    pthread_rwlock_init(_refcount_lock, lockattr_ref)
                    self._refcount_lock = _refcount_lock # don't set this attribute from None until we know the rwlock has been inited
                finally:
                    pthread_rwlockattr_destroy(lockattr_ref)
                with self.lock_refcount():
                    self._refcount_header.refcount = 1
                # finally, set the magic cookie saying that this thing is ready to go
                size_header.magic_cookie = self._MAGIC_COOKIE
        
            else:
                self.descr = bytes(data_layout.descr)
                self._refcount_lock = ctypes.byref(self._refcount_header.refcount_lock)
                with self.lock_refcount():
                    self._refcount_header.refcount += 1
            self._all_allocated = True
        except BaseException as e:
            # something failed somewhere in setting things up
            if create and hasattr(self, '_refcount_lock'):
                pthread_rwlock_destroy(self._refcount_lock)

            if hasattr(self, '_mmap_data'):
                # if we mmap'd data, munmap it
                munmap(self._mmap_data, self._mmap_size)

            if hasattr(self, '_fd'):
                # if we got an fd open, close it
                os.close(self._fd)
                if create:
                    shm_unlink(self._name)
            raise e


    @contextlib.contextmanager
    def lock_refcount(self):
        pthread_rwlock_wrlock(self._refcount_lock)
        yield
        pthread_rwlock_unlock(self._refcount_lock)

    def __del__(self):
        if not self._all_allocated:
            # in the case of incomplete init, the init function cleans up after itself
            return
        destroy = False
            
        # The refcount lock resides within the shared memory region to be destroyed and thus does not prevent
        # another thread in the same process from initiating a deep copy of this ISM object in the interval between
        # refcount lock release and completion of this destructor function. However, if this destructor is executing,
        # it is as a consequence of no Python references to this object remaining, precluding the possibility of a copy
        # operation being initiated except behind the interpreter's back. Therefore, a second in-process lock
        # to guard against this is not required. Additionally, if refcount reaches zero below, no other processes
        # are still referring to the shared memory, precluding the possibility of a copy operation in *another process*
        # during the interval between refLock destruction and completion of this destructor function, so a second
        # shared-process lock to guard against this is not required.
        with self.lock_refcount():
            self._refcount_header.refcount -= 1
            if self._refcount_header.refcount == 0:
                destroy = True
        if destroy:
            pthread_rwlock_destroy(self._refcount_lock)
        munmap(self._mmap_data, self._mmap_size)
        os.close(self._fd)
        if destroy:
            shm_unlink(self._name)

    @property
    def name(self):
        return self._name.decode('utf-8')

    @property
    def shared_refcount(self):
        with self.lock_refcount():
            return self._refcount_header.refcount

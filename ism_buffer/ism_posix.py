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
import mmap
import sys
import weakref

from . import ism_base

# Set up ctype types and wrappers for various system API functions.  The differences between the Linux and OS X
# calls and types that we require are slim, allowing both to share code after this section.

if sys.platform == 'linux':
    lib = ctypes.CDLL('librt.so.1', use_errno=True)
    # NB: 3rd argument, mode_t, is 4 bytes on linux and 2 bytes on osx (64 bit linux and osx, that is.  I
    # haven't had a chance to try this on 32-bit platforms, but patches / pull requests are welcome.)
    shm_open_argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint32]
    pthread_rwlockattr_t = ctypes.c_byte * 8
    pthread_rwlock_t = ctypes.c_byte * 56
elif sys.platform == 'darwin':
    lib = ctypes.CDLL('libc.dylib', use_errno=True)
    shm_open_argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint16]
    pthread_rwlockattr_t = ctypes.c_byte * 24
    pthread_rwlock_t = ctypes.c_byte * 200
else:
    raise NotImplementedError("ism_buffer's POSIX implementation is currently only for Linux and Darwin")

pthread_rwlockattr_t_p = ctypes.POINTER(pthread_rwlockattr_t)
pthread_rwlock_t_p = ctypes.POINTER(pthread_rwlock_t)
PTHREAD_PROCESS_SHARED = 1

def register_lib_func(func_name, argtypes, err='pthread'):
    func = getattr(lib, func_name)
    func.argtypes = argtypes
    if err == 'pthread':
        def errcheck(result, func, args):
            if result != 0:
                raise OSError(result, '{}(..) failed: {}'.format(func_name, describe_sys_errno(result)))
            return result
    elif err == 'os':        
        def errcheck(result, func, args):
            if result < 0:
                e = ctypes.get_errno()
                if e == 0:
                    raise RuntimeError(func_name + ' failed, but errno is 0.')
                raise OSError(e, func_name + ' failed: ' + describe_sys_errno(e))
            return result
    func.errcheck = errcheck

def describe_sys_errno(e):
    try:
        strerror = os.strerror(e)
    except ValueError:
        strerror = 'no description available'
    return '{} ({})'.format(strerror, errno.errorcode.get(e, 'UNKNOWN ERROR'))

API = [
    ('pthread_rwlock_destroy', [pthread_rwlock_t_p], 'pthread'),
    ('pthread_rwlock_init', [pthread_rwlock_t_p, pthread_rwlockattr_t_p], 'pthread'),
    ('pthread_rwlock_unlock', [pthread_rwlock_t_p], 'pthread'),
    ('pthread_rwlock_wrlock', [pthread_rwlock_t_p], 'pthread'),
    ('pthread_rwlockattr_destroy', [pthread_rwlockattr_t_p], 'pthread'),
    ('pthread_rwlockattr_init', [pthread_rwlockattr_t_p], 'pthread'),
    ('pthread_rwlockattr_setpshared', [pthread_rwlockattr_t_p, ctypes.c_int], 'pthread'),
    ('shm_open', shm_open_argtypes, 'os'),
    ('shm_unlink', [ctypes.c_char_p], 'os')
]

for func_name, argtypes, err in API:
    register_lib_func(func_name, argtypes, err)

# layout of ISMBuffer: SizeHeader, descr, data, RefCountHeader
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

@contextlib.contextmanager
def locking(lock):
    lib.pthread_rwlock_wrlock(lock)
    try:
        yield
    finally:
        lib.pthread_rwlock_unlock(lock)

class ISMBuffer(ism_base.ISMBase):
    def __init__(self, name, create=False, permissions=0o600, size=0, descr=b''):
        """Note: The default value for createPermissions, 0o600, or 384, represents the unix permission "readable/writeable
        by owner"."""
        super().__init__(name, create, permissions, size, descr)
        self._name = str(name).encode('utf-8') if type(name) is not bytes else name
        mmap_f, refcount_lock, fd = None, None, None # If an error happens in init, non-None values are an indication that these need to be cleaned up
        try:
            # First, figure out the sizes of everything. Easy if we're creating the buffer; requires a bit of digging if not.
            if create:
                self.size = size
                descr_size = len(descr)
            else:
                fd = lib.shm_open(self._name, os.O_RDWR, 0)
                with mmap.mmap(fd, ctypes.sizeof(SizeHeader), prot=mmap.PROT_READ) as size_header_mmap:
                    size_header = SizeHeader.from_buffer_copy(size_header_mmap)
                    assert size_header.magic_cookie == self._MAGIC_COOKIE
                    self.size = size_header.data_size
                    descr_size = size_header.descr_size
        
            class DataLayout(ctypes.Structure):
                _fields_ = [
                    ('size_header', SizeHeader),
                    ('descr', ctypes.c_uint8*descr_size),
                    ('data', ctypes.c_uint8*self.size),
                    ('refcount_header', RefCountHeader)
                ]
            mmap_size = ctypes.sizeof(DataLayout)
        
            if create:
                # If we're creating it, open the fd now. If it was extant, the fd already got opened above.
                fd = lib.shm_open(self._name, os.O_RDWR | os.O_CREAT | os.O_EXCL, permissions)
                os.ftruncate(fd, mmap_size)
            
            mmap_f = mmap.mmap(fd, mmap_size)
            data_layout = DataLayout.from_buffer(mmap_f)
            refcount_header = data_layout.refcount_header
            data = data_layout.data
            self.__array_interface__ = {
                'shape': (self.size,),
                'typestr':'|u1',
                'version':3,
                'data':(ctypes.addressof(data), False)
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
                lib.pthread_rwlockattr_init(lockattr_ref)
                try:
                    lib.pthread_rwlockattr_setpshared(lockattr_ref, PTHREAD_PROCESS_SHARED)
                    _refcount_lock = ctypes.byref(refcount_header.refcount_lock)
                    lib.pthread_rwlock_init(_refcount_lock, lockattr_ref)
                    refcount_lock = _refcount_lock # don't set this attribute until we know the rwlock has been inited
                finally:
                    lib.pthread_rwlockattr_destroy(lockattr_ref)
                with locking(refcount_lock):
                    refcount_header.refcount = 1
                # finally, set the magic cookie saying that this thing is ready to go
                size_header.magic_cookie = self._MAGIC_COOKIE
        
            else:
                self.descr = bytes(data_layout.descr)
                refcount_lock = ctypes.byref(refcount_header.refcount_lock)
                with locking(refcount_lock):
                    refcount_header.refcount += 1
            
            self._refcount_header = weakref.ref(refcount_header)
            # instead of having a __del__ method, we'll use weakref.finalize, which is called BOTH when the object is deleted
            # AND when the system exits.
            finalizer = Finalizer(self._name, refcount_header, mmap_f, fd)
            weakref.finalize(self, finalizer)
            
        except:
            # something failed somewhere in setting things up
            if create and refcount_lock is not None:
                lib.pthread_rwlock_destroy(refcount_lock)

            if mmap_f is not None:
                mmap_f.close()

            if fd is not None:
                # if we got an fd open, close it
                os.close(fd)
                if create:
                    lib.shm_unlink(self._name)
            raise
    
    @property
    def name(self):
        return self._name.decode('utf-8')

    @property
    def shared_refcount(self):
        refcount_header = self._refcount_header()
        if refcount_header is not None:
            with locking(ctypes.byref(refcount_header.refcount_lock)):
                return refcount_header.refcount

class Finalizer:    
    def __init__(self, name, refcount_header, mmap_f, fd):
        self.name = name
        self.refcount_header = refcount_header
        self.mmap_f = mmap_f
        self.fd = fd
    
    def __call__(self):
        destroy = False
        refcount_lock = ctypes.byref(self.refcount_header.refcount_lock)
        with locking(refcount_lock):
            self.refcount_header.refcount -= 1
            if self.refcount_header.refcount == 0:
                destroy = True        
        if destroy:
            lib.pthread_rwlock_destroy(refcount_lock)
        del self.refcount_header
        del refcount_lock
        self.mmap_f.close()
        os.close(self.fd)
        if destroy:
            lib.shm_unlink(self.name)

    
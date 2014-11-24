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

import ctypes

_CreateMutex = ctypes.windll.kernel32.CreateMutexW
# Note that the first parameter to CreateMutexW is actually a SECURITY_ATTRIBUTES* and not a void*.
# Providing nullptr for this parameter causes the mutex to inherit the creating process's DACL, which
# is typically what you want.
_CreateMutex.argtypes = [ctypes.wintypes.LPVOID, ctypes.wintypes.BOOL, ctypes.wintypes.LPWSTR]
_CreateMutex.restype = ctypes.wintypes.HANDLE
_WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
_WaitForSingleObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD]
_WaitForSingleObject.restype = ctypes.wintypes.DWORD
_ReleaseMutex = ctypes.windll.kernel32.ReleaseMutex
_ReleaseMutex.argtypes = [ctypes.wintypes.HANDLE]
_ReleaseMutex.restype = ctypes.wintypes.BOOL
_CreateFileMapping = ctypes.windll.kernel32.CreateFileMappingW
_CreateFileMapping.argtypes = [
    ctypes.wintypes.HANDLE,
    ctypes.wintypes.LPVOID, # In reality, another SECURITY_ATTRIBUTES* that defaults if given nullptr
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.LPWSTR
]
_CreateFileMapping.restype = ctypes.wintypes.HANDLE
_MapViewOfFile = ctypes.windll.kernel32.MapViewOfFile
_MapViewOfFile.argtypes = [
    ctypes.wintypes.HANDLE,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.c_size_t
]
_MapViewOfFile.restype = ctypes.wintypes.LPVOID
_ERROR_ALREADY_EXISTS = 183
_FILE_MAP_ALL_ACCESS = 0xF001F
_INVALID_HANDLE_VALUE = 0xFFFFFFFF
_PAGE_READWRITE = 0x04
_WAIT_TIMEOUT = 0x102
_WAIT_ABANDONED = 0x80

class _ISMBufferHeader(ctypes.Structure):
    '''On win32, first comes the struct, then comes the refCountLock mutex name (no null terminator - length is specified
    in header refCountMutexNameLength field).'''
    _fields_ = [
        ('refCount', ctypes.c_uint64),
        ('userSize', ctypes.c_uint64),
        ('refCountMutexNameLength', ctypes.c_uint16)
    ]

_ISMBufferHeader_p = ctypes.POINTER(_ISMBufferHeader)

raise NotImplementedError('win32 support is under construction [insert geocities_under_construction.gif here]')

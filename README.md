# Interprocess Shared Memory Buffer

Authors: Erik Hvatum and [Zachary Pincus](http://zplab.wustl.edu) <zpincus@gmail.com>

Simple library to create named shared-memory regions in Python 3 to transfer data between processes with no copying. Works on Linux and OS X, and probably other similar platforms.

In particular, the ISMBuffer code is optimized to share numpy arrays between processes, but arbitrary data can be stored.

The regions contain an internal reference count so are deleted when the last process is done with the region.
This simplifies the bookkeeping required for data sharing between processes.
The fact that the regions are named allows arbitrary processes to easily open the regions,
unlike anonymous shared-memory regions which are difficult to share except between forked processes.

import distutils.core
import pathlib
import numpy

distutils.core.setup(name = 'ism_buffer',
        version = '1.0',
        description = 'Tool to easily share memory buffers between processes',
        packages = ['ism_buffer'])

from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include
import sysconfig
from distutils.extension import Extension

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-DNDEBUG", "-O3", "-std=c11"]

ext_modules = [
    Extension('cymatrix',
              sources=["cymatrix.pyx", "src/C-Thread-Pool/thpool.c"],
              include_dirs=[
                  get_include(),
                  "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
                  "src/C-Thread-Pool"
              ],
              extra_compile_args=extra_compile_args),
    Extension('stlmap',
              sources=["stlmap.pyx"],
              include_dirs=[
                  get_include(),
              ],
              language="c++",
              extra_compile_args=extra_compile_args),
]

setup(name='cities',
      ext_modules=cythonize(ext_modules,
                            compiler_directives={'language_level': "3"}))

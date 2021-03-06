#setup.py
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import glob
import os
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy
from pathlib import Path

ext_modules=[]
for ext in Path('').glob('**/cython/resources/*.pyx'):
	ext = str(ext)
	filename = os.path.basename(ext)
	dirname = os.path.dirname(ext)
	aux = filename.split('.')
	if len(aux) > 2:
		continue
	name, _ = aux
	name = os.path.dirname(dirname) + '/' + name
	name = name.replace('/', '.')
	print(name)
	ext_modules.append(Extension(name, 
		sources=[ext],
		#extra_compile_args = ['-I/usr/local/opt/llvm/include', '-L/usr/local/opt/llvm/lib', '-O3', '-ffast-math', '-march=native', '-static', '-fopenmp=libomp'],
		extra_compile_args = ['-O3', '-ffast-math', '-march=native', '-static', '-fopenmp'],
		extra_link_args = ['-fopenmp'],
		library_dirs=['/usr/local/opt/libomp/lib'],
		include_dirs=['/usr/local/opt/libomp/include']
    ))

setup(
	name = 'cython_resources',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules,
	include_dirs=[numpy.get_include()],

)
#export CC=/usr/local/opt/llvm/bin/clang

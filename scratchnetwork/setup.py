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

ext_modules=[]
for ext in glob.glob("*/cython/resources/*.pyx"):
	filename = os.path.basename(ext)
	dirname = os.path.dirname(ext)
	aux = filename.split(".")
	if len(aux) > 2:
		continue
	name, _ = aux
	name = os.path.dirname(dirname) + "/" + name
	name = name.replace("/", ".")
	print(name)
	ext_modules.append(Extension(name, 
		sources=[ext],
		extra_compile_args = ["-I/usr/local/opt/llvm/include", "-L/usr/local/opt/llvm/lib", "-O3", "-ffast-math", "-march=native", '-static'], #-fopenmp=libomp
		#extra_link_args
    ))

setup(
	name = 'cython_resources',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules,
	include_dirs=[numpy.get_include()],
)
#export CC=/usr/local/opt/llvm/bin/clang
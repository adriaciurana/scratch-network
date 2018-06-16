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
	name, _ = filename.split(".")
	name = os.path.dirname(dirname) + "/" + name
	name = name.replace("/", ".")
	print(name)
	print(ext)
	ext_modules.append(Extension(name, [ext]))

setup(
	name = 'cython_resources',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules,
	include_dirs=[numpy.get_include()],
	extra_compile_args=['-o3', '-Wno-#warnings']
)
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


geometric_utils = Extension(
	"pybalu.feature_extraction.geometric_utils",
	sources=["pybalu/feature_extraction/geometric_utils.pyx"],
	include_dirs=[numpy.get_include()],
)

hog_utils = Extension(
	"pybalu.feature_extraction.hog_utils",
	sources=["pybalu/feature_extraction/hog_utils.pyx"],
	include_dirs=[numpy.get_include()],
)

flusser_utils = Extension(
	"pybalu.feature_extraction.flusser_utils",
	sources=["pybalu/feature_extraction/flusser_utils.pyx"],
	include_dirs=[numpy.get_include()],
)

haralick_utils = Extension(
	"pybalu.feature_extraction.haralick_utils",
	sources=["pybalu/feature_extraction/haralick_utils.pyx"],
	include_dirs=[numpy.get_include()],
)

external_modules = cythonize(
	[
		geometric_utils,
		hog_utils,
		flusser_utils,
		haralick_utils,
	],
	compiler_directives={"language_level": 3},
)

setup(
	ext_modules=external_modules,
)

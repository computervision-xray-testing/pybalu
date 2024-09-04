import os
import numpy
import shutil
from setuptools import Distribution, Extension

from Cython.Build import build_ext, cythonize


cython_dir = "pybalu"
output_dir = "cython_build"

# Manage extensions
geometric_utils = Extension(
	"pybalu.feature_extraction.geometric_utils",
	sources=[os.path.join(cython_dir, "feature_extraction", "geometric_utils.pyx")],
	include_dirs=[numpy.get_include()],
	extra_compile_args=["-O3"],
)

hog_utils = Extension(
	"pybalu.feature_extraction.hog_utils",
	sources=[os.path.join(cython_dir, "feature_extraction", "hog_utils.pyx")],
	include_dirs=[numpy.get_include()],
	extra_compile_args=["-O3"],
)

flusser_utils = Extension(
	"pybalu.feature_extraction.flusser_utils",
	sources=[os.path.join(cython_dir, "feature_extraction", "flusser_utils.pyx")],
	include_dirs=[numpy.get_include()],
	extra_compile_args=["-O3"],
)

haralick_utils = Extension(
	"pybalu.feature_extraction.haralick_utils",
	sources=[os.path.join(cython_dir, "feature_extraction", "haralick_utils.pyx")],
	include_dirs=[numpy.get_include()],
	extra_compile_args=["-O3"],
)

cython_modules = [geometric_utils, hog_utils, flusser_utils, haralick_utils]

external_modules = cythonize(
	cython_modules,
	compiler_directives={"language_level": 3},
	include_path=[cython_dir],
	build_dir=output_dir,
	annotate=False,
	force=True,
)

dist = Distribution({"ext_modules": external_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
	relative_extension = os.path.relpath(output, cmd.build_lib)
	shutil.copyfile(output, relative_extension)

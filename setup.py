# -*- coding: utf-8 -*-
#
import ast
from setuptools.extension import Extension
from setuptools import setup
import os
import sys
"""setuptools-based setup.py template for Cython projects.

Main setup for the library.

Usage as usual with setuptools:
    python setup.py build_ext
    python setup.py build
    python setup.py install
    python setup.py sdist

For details, see
    http://setuptools.readthedocs.io/en/latest/setuptools.html#command-reference
or
    python setup.py --help
    python setup.py --help-commands
    python setup.py --help bdist_wheel  # or any command
"""

#########################################################
# General config
#########################################################

# Name of the top-level package of your library.
#
# This is also the top level of its source tree, relative to the top-level project directory setup.py resides in.
#
libname = "pybalu"

# Choose build type.
#
build_type = "optimized"
# build_type="debug"

# Short description for package list on PyPI
#
SHORTDESC = "Python3 implementation of Computer Vision and Pattern Recognition library Balu"

# Long description for package homepage on PyPI
#
DESC = """Python3 implementation of Computer Vision and Pattern Recognition library Balu.

Made By Domingo Mery.
Implementation on Python3 done by Marco Bucchi.
"""

# Set up data files for packaging.
#
# Directories (relative to the top-level directory where setup.py resides) in which to look for data files.
datadirs = ("pybalu",)

# File extensions to be considered as data files. (Literal, no wildcards.)
dataexts = (".py",  ".pyx", ".pxd",  ".c", ".cpp", ".h",
            ".sh",  ".lyx", ".tex", ".txt", ".pdf")

# Standard documentation to detect (and package if it exists).
#
standard_docs = ["README", "LICENSE", "TODO", "CHANGELOG",
                 "AUTHORS"]  # just the basename without file extension
# commonly .md for GitHub projects, but other projects may use .rst or .txt (or even blank).
standard_doc_exts = [".md", ".rst", ".txt", ""]


#########################################################
# Init
#########################################################

# check for Python 3.6 or later
# http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

IS_WINDOWS = sys.platform.startswith("win")


USE_CYTHON = 'auto'             # True, 'auto' or False

if USE_CYTHON:
    try:
        from Cython.Build import cythonize
    except ImportError:
        if USE_CYTHON == 'auto':
            USE_CYTHON = False
        else:
            sys.exit(
                "Cython not found. Cython is needed to build the extension modules.")


#########################################################
# Definitions
#########################################################

# Define our base set of compiler and linker flags.
#
# This is geared toward x86_64, see
#    https://gcc.gnu.org/onlinedocs/gcc-4.6.4/gcc/i386-and-x86_002d64-Options.html
#
# Customize these as needed.
#
# Note that -O3 may sometimes cause mysterious problems, so we limit ourselves to -O2.

# Modules involving numerical computations
#
extra_compile_args_math_optimized = [
    '-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
extra_compile_args_math_debug = ['-march=native', '-O0', '-g']
extra_link_args_math_optimized = []
extra_link_args_math_debug = []

# Modules that do not involve numerical computations
#
extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug = ['-O0', '-g']
extra_link_args_nonmath_optimized = []
extra_link_args_nonmath_debug = []

# Additional flags to compile/link with OpenMP
#
openmp_compile_args = ['-fopenmp']
openmp_link_args = ['-fopenmp']


#########################################################
# Helpers
#########################################################

# Make absolute cimports work.
#
# See
#     https://github.com/cython/cython/wiki/PackageHierarchy
#
# For example: my_include_dirs = [np.get_include()]
my_include_dirs = ["."]
try:
    import numpy as np
    my_include_dirs.append(np.get_include())
except ModuleNotFoundError:
    pass


# Choose the base set of compiler and linker flags.
#
if build_type == 'optimized':
    my_extra_compile_args_math = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args_math = extra_link_args_math_optimized
    my_extra_link_args_nonmath = extra_link_args_nonmath_optimized
    my_debug = False
    print("build configuration selected: optimized")
elif build_type == 'debug':
    my_extra_compile_args_math = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args_math = extra_link_args_math_debug
    my_extra_link_args_nonmath = extra_link_args_nonmath_debug
    my_debug = True
    print("build configuration selected: debug")
else:
    raise ValueError(
        "Unknown build configuration '%s'; valid: 'optimized', 'debug'" % (build_type))


def declare_cython_extension(extName, use_math=False, use_openmp=False, include_dirs=None):
    """Declare a Cython extension module for setuptools.

Parameters:
    extName : str
        Absolute module name, e.g. use `mylibrary.mypackage.mymodule`
        for the Cython source file `mylibrary/mypackage/mymodule.pyx`.

    use_math : bool
        If True, set math flags and link with ``libm``.

    use_openmp : bool
        If True, compile and link with OpenMP.

Return value:
    Extension object
        that can be passed to ``setuptools.setup``.
"""
    extPath = extName.replace(".", os.path.sep)

    if USE_CYTHON:
        extPath += ".pyx"
    else:
        extPath += ".c"

    libraries = None

    if use_math:
        compile_args = list(my_extra_compile_args_math)  # copy
        link_args = list(my_extra_link_args_math)
        # link libm; this is a list of library names without the "lib" prefix
        if not IS_WINDOWS:
            libraries = ["m"]
    else:
        compile_args = list(my_extra_compile_args_nonmath)
        link_args = list(my_extra_link_args_nonmath)
        libraries = None  # value if no libraries, see setuptools.extension._Extension

    # OpenMP
    if use_openmp:
        compile_args.insert(0, openmp_compile_args)
        link_args.insert(0, openmp_link_args)

    # See
    #    http://docs.cython.org/src/tutorial/external.html
    #
    # on linking libraries to your Cython extensions.
    #
    return Extension(extName,
                     [extPath],
                     extra_compile_args=compile_args,
                     extra_link_args=link_args,
                     include_dirs=include_dirs,
                     libraries=libraries
                     )


# Gather user-defined data files
#
# http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
#
datafiles = []


def getext(filename): return os.path.splitext(filename)[1]


for datadir in datadirs:
    datafiles.extend([(root, [os.path.join(root, f) for f in files if getext(f) in dataexts])
                      for root, dirs, files in os.walk(datadir)])


# Add standard documentation (README et al.), if any, to data files
#
detected_docs = []
for docname in standard_docs:
    for ext in standard_doc_exts:
        # relative to the directory in which setup.py resides
        filename = "".join((docname, ext))
        if os.path.isfile(filename):
            detected_docs.append(filename)
datafiles.append(('.', detected_docs))


# Extract __version__ from the package __init__.py
# (since it's not a good idea to actually run __init__.py during the build process).
#
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
#
init_py_path = os.path.join(libname, '__init__.py')
version = 'ERROR'
try:
    with open(init_py_path) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            print("WARNING: Version information not found in '%s', using placeholder '%s'" % (
                init_py_path, version), file=sys.stderr)
except FileNotFoundError:
    print("WARNING: Could not find file '%s', using placeholder version information '%s'" % (
        init_py_path, version), file=sys.stderr)

#########################################################
# Set up modules
#########################################################

# declare Cython extension modules here
#
# ext_module_feature_analysis = declare_cython_extension( "pybalu.feature_analysis", use_math=True, include_dirs=my_include_dirs)
ext_module_fex_flusser = declare_cython_extension(
    "pybalu.feature_extraction.flusser_utils", use_math=True, include_dirs=my_include_dirs)
ext_module_fex_geometric = declare_cython_extension(
    "pybalu.feature_extraction.geometric_utils", use_math=True, include_dirs=my_include_dirs)
ext_module_fex_haralick = declare_cython_extension(
    "pybalu.feature_extraction.haralick_utils", use_math=True, include_dirs=my_include_dirs)
ext_module_fex_hog = declare_cython_extension(
    "pybalu.feature_extraction.hog_utils", use_math=True, include_dirs=my_include_dirs)

# this is mainly to allow a manual logical ordering of the declared modules
#
cython_ext_modules = [ext_module_fex_flusser,
                      ext_module_fex_geometric,
                      ext_module_fex_haralick,
                      ext_module_fex_hog]

# Call cythonize() explicitly, as recommended in the Cython documentation. See
#     http://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiling-with-distutils
#
# This will favor Cython's own handling of '.pyx' sources over that provided by setuptools.
#
# Note that my_ext_modules is just a list of Extension objects. We could add any C sources (not coming from Cython modules) here if needed.
# cythonize() just performs the Cython-level processing, and returns a list of Extension objects.
#
if USE_CYTHON:
    my_ext_modules = cythonize(
        cython_ext_modules, include_path=my_include_dirs, gdb_debug=my_debug)
else:
    my_ext_modules = cython_ext_modules


#########################################################
# Call setup()
#########################################################

setup(
    name="pybalu",
    version=version,
    author="Marco Bucchi",
    author_email="mabucchi@uc.cl",
    url="https://github.com/mbucchi",

    description=SHORTDESC,
    long_description=DESC,

    # CHANGE THIS
    license="MIT",

    # free-form text field; http://stackoverflow.com/questions/34994130/what-platforms-argument-to-setup-in-setup-py-does
    platforms=["Linux"],

    # See
    #    https://pypi.python.org/pypi?%3Aaction=list_classifiers
    #
    # for the standard classifiers.
    #
    # Remember to configure these appropriately for your project, especially license!
    #
    classifiers=["Development Status :: 3 - Alpha",
                 "Environment :: Console",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 # not a standard classifier; CHANGE THIS
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: POSIX :: Linux",
                 "Programming Language :: Cython",
                 "Programming Language :: Python",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.6",
                 "Topic :: Scientific/Engineering",
                 "Topic :: Scientific/Engineering :: Mathematics",
                 ],

    # See
    #    http://setuptools.readthedocs.io/en/latest/setuptools.html
    #
    install_requires=[
        "numpy>=1.16.1",
        "scipy>=1.1.0",
        "imageio>=2.5.0",
        "Pillow>=7.2.0",
        "scikit-image>=0.17.2",
        "scikit-learn>=0.22.2",
        "tqdm>=4.29.1"
    ],
    setup_requires=[
        "cython>=0.29.6",
        "numpy>=1.16.1",
        "scipy>=1.1.0",
        "imageio>=2.5.0",
        "Pillow>=7.2.0",
        "scikit-image>=0.17.2",
        "scikit-learn>=0.22.2",
        "tqdm>=4.29.1"
    ],
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*, <4',
    provides=["pybalu"],

    # keywords for PyPI (in case you upload your project)
    #
    # e.g. the keywords your project uses as topics on GitHub, minus "python" (if there)
    #
    keywords=["image pattern recognition computer vision"],

    # All extension modules (list of Extension objects)
    #
    ext_modules=my_ext_modules,

    # Declare packages so that  python -m setup build  will copy .py files (especially __init__.py).
    #
    # This **does not** automatically recurse into subpackages, so they must also be declared.
    #
    packages=["pybalu",
              "pybalu.classification",
              "pybalu.data_selection",
              "pybalu.feature_analysis",
              "pybalu.feature_extraction",
              "pybalu.feature_selection",
              "pybalu.feature_transformation",
              "pybalu.img_processing",
              "pybalu.io",
              "pybalu.misc",
              "pybalu.performance_eval"],

    # Install also Cython headers so that other Cython modules can cimport ours
    #
    # Fileglobs relative to each package, **does not** automatically recurse into subpackages.
    #
    # FIXME: force sdist, but sdist only, to keep the .c files (this puts them also in the bdist)
    package_data={'pybalu': ['*.c'],
                  'pybalu.classification': ['*.c'],
                  'pybalu.data_selection': ['*.c'],
                  'pybalu.feature_analysis': ['*.c'],
                  'pybalu.feature_extraction': ['*.c'],
                  'pybalu.feature_selection': ['*.c'],
                  'pybalu.feature_transformation': ['*.c'],
                  'pybalu.img_processing': ['*.c'],
                  'pybalu.io': ['*.c'],
                  'pybalu.misc': ['*.c'],
                  'pybalu.performance_eval': ['*.c'],
                  'pybalu.utils': ['*.c']},

    # Disable zip_safe, because:
    #   - Cython won't find .pxd files inside installed .egg, hard to compile libs depending on this one
    #   - dynamic loader may need to have the library unzipped to a temporary directory anyway (at import time)
    #
    zip_safe=False,

    # Custom data files not inside a Python package
    data_files=datafiles
)

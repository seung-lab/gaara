project(
  'fastgaara',
  'cpp', 
  version: '0.7.0',
  default_options: ['cpp_std=c++20'],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

pybind11_dep = dependency('pybind11')
pybind11_incdir = run_command(py, ['-c', 'import pybind11; print(pybind11.get_include())'], check: true).stdout().strip()

numpy_incdir = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()

src_files = files('gaara/fastgaara.cpp')

# install_data(
#   ['lookup_tables/tables/simple.bin', 'lookup_tables/tables/isthmus.bin'],
#   install_dir: py.get_install_dir() / 'lookup_tables' / 'tables',
# )

py.install_sources(
  'gaara/__init__.py',
  subdir: 'gaara',
)

target = py.extension_module(
  'fastgaara',
  src_files,
  dependencies: [py_dep, pybind11_dep],
  include_directories: [
    include_directories('src'), 
    include_directories(numpy_incdir),
    include_directories(pybind11_incdir),
  ],
  cpp_args: [
    '-DGAARA_STATICALLY_LINK_LUTS',
    '-mmacosx-version-min=11.0',
  ],
  install: true,
  subdir: 'gaara',
)
project('fastgaara', 'cpp', version: '0.1.0')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

pybind11_dep = dependency('pybind11')
pybind11_incdir = run_command(py, ['-c', 'import pybind11; print(pybind11.get_include())'], check: true).stdout().strip()

numpy_incdir = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()

src_files = files('gaara/fastgaara.cpp')

target = py.extension_module(
  'fastgaara',
  src_files,
  dependencies: [py_dep, pybind11_dep],
  include_directories: [
    include_directories('src'), 
    include_directories(numpy_incdir),
    include_directories(pybind11_incdir),
  ],
  install: true,
)
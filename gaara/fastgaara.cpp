#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cmath>

#include "gaara.hpp"

namespace py = pybind11;

// assumes fortran order
py::array thin_palagyi(const py::array& labels) {
	py::dtype dt = labels.dtype();
	int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

	uint8_t* labels_ptr = const_cast<uint8_t*>(labels.data());

	gaara::thin_palagyi(labels_ptr, sx, sy, sz);

	return labels;
}

PYBIND11_MODULE(fastgaara, m) {
	m.doc() = "Python interface for Gaara C++ functions."; 
	m.def("thin_palagyi", &thin_palagyi, "Perform morphological thinning using the Palagyi algorithm on a 3D image.");
}
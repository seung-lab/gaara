#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cmath>

#include "gaara_binary.hpp"
#include "gaara_multilabel.hpp"

namespace py = pybind11;

// assumes fortran order
py::array thin_palagyi_binary(const py::array& labels) {
	py::dtype dt = labels.dtype();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

	uint8_t* labels_ptr = static_cast<uint8_t*>(const_cast<void*>(labels.data()));

	gaara::binary::skeletonize(labels_ptr, sx, sy, sz);

	return labels;
}

// assumes fortran order
py::array thin_palagyi_multilabel(const py::array& labels) {
	py::dtype dt = labels.dtype();
	int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

#define SKELETONIZE(data_type)\
	gaara::multilabel::skeletonize<data_type>(\
		static_cast<data_type*>(const_cast<void*>(labels.data())),\
		sx, sy, sz\
	);

	if (width == 1) {
		SKELETONIZE(uint8_t)
	}
	else if (width == 2) {
		SKELETONIZE(uint16_t)
	}
	else if (width == 4) {
		SKELETONIZE(uint32_t)
	}
	else {
		SKELETONIZE(uint64_t)
	}

#undef SKELETONIZE

	return labels;
}

PYBIND11_MODULE(fastgaara, m) {
	m.doc() = "Python interface for Gaara C++ functions."; 
	m.def("thin_palagyi_binary", &thin_palagyi_binary, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image.");
	m.def("thin_palagyi_multilabel", &thin_palagyi_multilabel, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image.");
}
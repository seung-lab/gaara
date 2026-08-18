#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cmath>

#include "gaara_binary.hpp"
#include "gaara_multilabel.hpp"

namespace py = pybind11;

template <typename Func>
py::array dispatch_skeletonize(const py::array& labels, Func&& func) {
	py::dtype dt = labels.dtype();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

	void* data = const_cast<void*>(labels.data());

	if (width == 1) {
		return func(static_cast<uint8_t*>(data));
	}
	else if (width == 2) {
		return func(static_cast<uint16_t*>(data));
	}
	else if (width == 4) {
		return func(static_cast<uint32_t*>(data));
	}
	else {
		return func(static_cast<uint64_t*>(data));
	}
}

// assumes fortran order
py::array thin_palagyi_binary(const py::array& labels) {
	return dispatch_skeletonize(labels, [](auto *data, uint64_t sx, uint64_t, sy, uint64_t sz) {
		return gaara::binary::skeletonize<decltype(data)>(
			static_cast<data_type*>(const_cast<void*>(labels.data())),
			sx, sy, sz
		);
	});
}

// assumes fortran order
py::array thin_palagyi_multilabel(const py::array& labels) {
	return dispatch_skeletonize(labels, [](auto *data, uint64_t sx, uint64_t, sy, uint64_t sz) {
		return gaara::multilabel::skeletonize<decltype(data)>(
			static_cast<data_type*>(const_cast<void*>(labels.data())),
			sx, sy, sz
		);
	});
}

PYBIND11_MODULE(fastgaara, m) {
	m.doc() = "Python interface for Gaara C++ functions."; 
	m.def("thin_palagyi_binary", &thin_palagyi_binary, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image.");
	m.def("thin_palagyi_multilabel", &thin_palagyi_multilabel, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image.");
}
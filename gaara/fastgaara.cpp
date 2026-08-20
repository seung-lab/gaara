#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdlib>
#include <list>
#include <unordered_map>

#include "defs.hpp"
#include "gaara_binary.hpp"
#include "gaara_multilabel.hpp"

namespace py = pybind11;

template <typename Func>
auto dispatch_skeletonize(const py::array& labels, Func&& func) {
	py::dtype dt = labels.dtype();
	const int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

	void* data = const_cast<void*>(labels.data());

	if (width == 1) {
		func(static_cast<uint8_t*>(data), sx, sy, sz);
	}
	else if (width == 2) {
		func(static_cast<uint16_t*>(data), sx, sy, sz);
	}
	else if (width == 4) {
		func(static_cast<uint32_t*>(data), sx, sy, sz);
	}
	else {
		func(static_cast<uint64_t*>(data), sx, sy, sz);
	}
}

// assumes fortran order
py::array thin_palagyi_binary(const py::array& labels) {
	dispatch_skeletonize(labels, [](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
		using T = std::remove_pointer_t<decltype(data)>;
		return gaara::binary::skeletonize<T>(
			data,
			sx, sy, sz
		);
	});

	return labels;
}

// assumes fortran order
py::array thin_palagyi_multilabel(const py::array& labels) {
	dispatch_skeletonize(labels, [](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
		using T = std::remove_pointer_t<decltype(data)>;
		return gaara::multilabel::skeletonize<T>(
			data,
			sx, sy, sz
		);
	});

	return labels;
}

PYBIND11_MODULE(fastgaara, m) {
	m.doc() = "Python interface for Gaara C++ functions."; 
	m.def("thin_palagyi_binary", &thin_palagyi_binary, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image.");
	m.def("thin_palagyi_multilabel", &thin_palagyi_multilabel, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image.");
}
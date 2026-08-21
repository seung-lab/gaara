#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdlib>
#include <list>
#include <unordered_map>

#include "def.hpp"
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
		return func(static_cast<uint8_t*>(data), sx, sy, sz);
	}
	else if (width == 2) {
		return func(static_cast<uint16_t*>(data), sx, sy, sz);
	}
	else if (width == 4) {
		return func(static_cast<uint32_t*>(data), sx, sy, sz);
	}
	else {
		return func(static_cast<uint64_t*>(data), sx, sy, sz);
	}
}

py::array vertices_to_numpy(
	const gaara::def::Skeleton& skel,
	const uint64_t sx, const uint64_t sy
) {
	const uint64_t sxy = sx * sy;
	std::vector<py::ssize_t> vertex_shape = {
		static_cast<py::ssize_t>(skel.vertices.size()),
		3
	};
	py::array_t<float> vertices(vertex_shape);

	auto vmut = vertices.mutable_unchecked<2>();

	for (uint64_t i = 0; i < skel.vertices.size(); i++) {
		const uint64_t vert = skel.vertices[i];
		const uint64_t z = vert / sxy;
		const uint64_t y = (vert - z * sxy) / sx;
		const uint64_t x = vert - z * sxy - sx * y;
		vmut(i,0) = x;
		vmut(i,1) = y;
		vmut(i,2) = z;
	}

	return vertices;
}

py::array edges_to_numpy(const gaara::def::Skeleton& skel) {
	std::vector<py::ssize_t> edge_shape = {
		static_cast<py::ssize_t>(skel.edges.size()),
		2
	};
	py::array_t<uint32_t> edges(edge_shape);

	auto mut = edges.mutable_unchecked<2>();

	for (uint64_t i = 0; i < skel.edges.size(); i++) {
		mut(i,0) = skel.edges[i].first;
		mut(i,1) = skel.edges[i].second;
	}

	return edges;
}

// assumes fortran order
void thin_binary(const py::array& labels, const bool preserve_endpoints) {
	dispatch_skeletonize(labels, 
		[preserve_endpoints](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
			using T = std::remove_pointer_t<decltype(data)>;
			return gaara::binary::thin<T>(
				data,
				sx, sy, sz,
				preserve_endpoints
			);
		}
	);
}

// assumes fortran order
void thin_multilabel(const py::array& labels, const bool preserve_endpoints) {
	dispatch_skeletonize(labels, 
		[preserve_endpoints](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
			using T = std::remove_pointer_t<decltype(data)>;
			return gaara::multilabel::thin<T>(
				data,
				sx, sy, sz,
				preserve_endpoints
			);
		}
	);
}

// assumes fortran order
auto thin_crackle(const py::buffer& buffer, const bool preserve_endpoints) {
	py::buffer_info info = buffer.request();

	if (info.ndim != 1) {
		throw std::runtime_error("Expected a 1D buffer");
	}

	std::span<unsigned char> data(info.ptr, info.size);
	return gaara::binary::thin_crackle(data, preserve_endpoints);
}

// assumes fortran order
auto skeletonize_binary(const py::array& labels, const bool preserve_endpoints) {
	auto skel = dispatch_skeletonize(labels, 
		[preserve_endpoints](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
			using T = std::remove_pointer_t<decltype(data)>;
			return gaara::binary::skeletonize<T>(
				data,
				sx, sy, sz,
				preserve_endpoints
			);
		}
	);

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];

	py::array vertices = vertices_to_numpy(skel, sx, sy);
	py::array edges = edges_to_numpy(skel);

	return py::make_tuple(vertices, edges);
}

// assumes fortran order
auto skeletonize_multilabel(const py::array& labels, const bool preserve_endpoints) {
	auto skeletons = dispatch_skeletonize(labels, 
		[preserve_endpoints](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
			using T = std::remove_pointer_t<decltype(data)>;
			return gaara::multilabel::skeletonize<T>(
				data,
				sx, sy, sz,
				preserve_endpoints
			);
		}
	);

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];

	py::dict py_skeletons;
	for (const auto& [segid, skel] : skeletons) {

		py::array vertices = vertices_to_numpy(skel, sx, sy);
		py::array edges = edges_to_numpy(skel);

		py_skeletons[py::int_(segid)] = py::make_tuple(vertices, edges);
	}

	return py_skeletons;
}

PYBIND11_MODULE(fastgaara, m) {
	m.doc() = "Python interface for Gaara C++ functions."; 
	m.def("thin_binary", &thin_binary, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image.");
	m.def("thin_multilabel", &thin_multilabel, "Perform morphological thinning using the Palagyi algorithm on a multilabel 3D image.");
	m.def("thin_crackle", &thin_crackle, "Perform morphological thinning using the Palagyi algorithm on crackle array.");
	m.def("skeletonize_binary", &skeletonize_binary, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image and convert to skeletons.");
	m.def("skeletonize_multilabel", &skeletonize_multilabel, "Perform morphological thinning using the Palagyi algorithm on a multilabel 3D image and convert to skeletons.");
}
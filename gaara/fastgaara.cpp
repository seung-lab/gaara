#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdlib>
#include <list>
#include <unordered_map>

#include "def.hpp"
#include "binary.hpp"
#include "multilabel.hpp"
#include "postprocess.hpp"

namespace py = pybind11;

template <typename Func>
auto dispatch(const py::array& labels, Func&& func) {
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
auto thin_binary(
	const py::array& labels, 
	const bool preserve_endpoints, 
	const py::array_t<uint16_t>& preserve_coords,
	const int64_t max_iterations = -1
) {
	const uint64_t Npc = preserve_coords.shape()[0];

	auto pc = preserve_coords.unchecked<2>();
	std::vector<gaara::def::Voxel> coords;
	coords.reserve(Npc);
	for (uint64_t i = 0; i < Npc; i++) {
		coords.emplace_back(pc(i,0), pc(i,1), pc(i,2));
	}

	return dispatch(labels, 
		[=](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
			using T = std::remove_pointer_t<decltype(data)>;
			return gaara::binary::thin<T>(
				data,
				sx, sy, sz,
				preserve_endpoints,
				coords,
				max_iterations
			);
		}
	);
}

// assumes fortran order
auto thin_multilabel(
	const py::array& labels,
	const bool preserve_endpoints,
	const int64_t max_iterations
) {
	return dispatch(labels, 
		[=](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
			using T = std::remove_pointer_t<decltype(data)>;
			return gaara::multilabel::thin<T>(
				data,
				sx, sy, sz,
				preserve_endpoints,
				max_iterations
			);
		}
	);
}

// assumes fortran order
auto skeletonize_binary(const py::array& labels, const bool preserve_endpoints) {
	auto skel = dispatch(labels, 
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
	auto skeletons = dispatch(labels, 
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

auto extract_skeletons(const py::array& labels) {
	auto skeletons = dispatch(labels, 
		[=](auto *data, uint64_t sx, uint64_t sy, uint64_t sz) {
			using T = std::remove_pointer_t<decltype(data)>;
			return gaara::postprocess::extract_skeletons<T>(
				data,
				sx, sy, sz
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
	m.def("skeletonize_binary", &skeletonize_binary, "Perform morphological thinning using the Palagyi algorithm on a binary 3D image and convert to skeletons.");
	m.def("skeletonize_multilabel", &skeletonize_multilabel, "Perform morphological thinning using the Palagyi algorithm on a multilabel 3D image and convert to skeletons.");
	m.def("extract_skeletons", &extract_skeletons, "Given a thinned image, extract the skeletons. This is just the second step broken out.");
}
#ifndef __GAARA_POSTPROCESS_HPP__
#define __GAARA_POSTPROCESS_HPP__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "def.hpp"

namespace gaara::postprocess {

using Edges = std::vector<std::pair<uint64_t, uint64_t>>;

std::vector<uint64_t> unique_vertices(const Edges& edges) {
	if (edges.size() == 0) {
		return std::vector<uint64_t>();
	}

	std::vector<uint64_t> all_vertices;
	all_vertices.reserve(edges.size() * 2);

	for (auto& edge : edges) {
		all_vertices.push_back(edge.first);
		all_vertices.push_back(edge.second);
	}

	std::sort(all_vertices.begin(), all_vertices.end());

	std::vector<uint64_t> uniq;

	uint64_t last = all_vertices[0]; // guaranteed to be at least 2
	uniq.push_back(last);

	for (uint64_t i = 1; i < all_vertices.size(); i++) {
		if (all_vertices[i] != last) {
			uniq.push_back(all_vertices[i]);
			last = all_vertices[i];
		}
	}

	return uniq;
}

template <typename LABEL>
Edges connected_component(
	LABEL* field, 
	const size_t sx, const size_t sy, const size_t sz, 
	const size_t source,
	std::vector<bool>& visited
) {
	Edges edges;

	if (field == nullptr) {
		return edges;
	}

	const size_t voxels = sx * sy * sz;	
	const size_t sxy = sx * sy;

	if (source >= voxels) {
		return edges;
	}
	else if (field[source] == 0) {
		return edges;
	}

	const LABEL label = field[source];

	std::deque<gaara::def::Voxel> stack;

	uint64_t zi = source / sxy;
	uint64_t yi = (source - zi * sxy) / sx;
	uint64_t xi = (source - zi * sxy - yi * sx);

	stack.emplace_back(xi,yi,zi);

	while (stack.size()) {
		gaara::def::Voxel point = stack.back();
		stack.pop_back();
		
		const int64_t loc = (
			  (int64_t)point.x 
			+ (int64_t)(sx * (uint64_t)point.y)
			+ (int64_t)(sxy * (uint64_t)point.z)
		);

		if (visited[loc]) {
			continue;
		}

		const int64_t minx = point.x > 0 ? -1 : 0;
		const int64_t miny = point.y > 0 ? -1 : 0;
		const int64_t minz = point.z > 0 ? -1 : 0;

		const int64_t maxx = point.x < sx - 1 ? 1 : 0;
		const int64_t maxy = point.y < sy - 1 ? 1 : 0;
		const int64_t maxz = point.z < sz - 1 ? 1 : 0;

		for (int64_t z = minz; z <= maxz; z++) {
			for (int64_t y = miny; y <= maxy; y++) {
				for (int64_t x = minx; x <= maxx; x++) {
					if (x == 0 && y == 0 && z == 0) {
						continue;
					}

					const int64_t neighboridx = loc + (x + sx * y + sxy * z);

					// Can't check visited here because there
					// could be real loops in the structure that
					// must be recorded. We'll just have to do
					// duplicate elimination after.
					if (field[neighboridx] == label) {
						stack.emplace_back(
							(int)point.x + (int)x,
							(int)point.y + (int)y,
							(int)point.z + (int)z
						);

						if (loc < neighboridx) {
							edges.emplace_back(loc, neighboridx);
						}
					}
				}
			}
		}

		visited[loc] = true;
	}

	return edges;
}

template <typename LABEL>
std::unordered_map<uint64_t, gaara::def::Skeleton> 
extract_skeleton(
	LABEL* labels,
	const uint64_t sx, const uint64_t sy, const uint64_t sz
) {
	const uint64_t voxels = sx * sy * sz;

	std::vector<bool> visited(voxels);

	using Edges = std::vector<std::pair<uint64_t, uint64_t>>;

	std::unordered_map<uint64_t, Edges> all_edges;
	std::unordered_map<uint64_t, gaara::def::Skeleton> all_skeletons;

	if (voxels == 0) {
		return all_skeletons;
	}

	for (uint64_t loc = 0; loc < voxels; loc++) {
		if (labels[loc] == 0 || visited[loc]) {
			continue;
		}

		Edges component_edges = \
			gaara::postprocess::connected_component(
				labels,
				sx, sy, sz,
				loc,
				visited
			);

		Edges& label_edges = all_edges[labels[loc]];

		label_edges.insert(
			label_edges.end(), component_edges.begin(), component_edges.end()
		);
	}

	for (auto [label, edges] : all_edges) {
		std::vector<uint64_t> uniq = gaara::postprocess::unique_vertices(edges);
		std::unordered_map<uint64_t, uint64_t> mapping;
		mapping.reserve(uniq.size());

		for (uint64_t i = 0; i < uniq.size(); i++) {
			mapping[uniq[i]] = i;
		}

		for (auto& edge : edges) {
			edge.first = mapping[edge.first];
			edge.second = mapping[edge.second];
		}

		all_skeletons[label] = gaara::def::Skeleton(uniq, edges);
	}

	return all_skeletons;
}

}; 

#endif
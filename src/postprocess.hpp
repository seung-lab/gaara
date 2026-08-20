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
#include <vector>

#include "def.hpp"

namespace gaara::postprocess {

template <typename LABEL>
std::vector<std::pair<uint64_t, uint64_t>> connected_component(
	LABEL* field, 
	const size_t sx, const size_t sy, const size_t sz, 
	const size_t source
) {
	std::vector<std::pair<uint64_t,uint64_t>> edges;

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

	std::vector<bool> visited(voxels);

	std::deque<gaara::def::Voxel> stack;
	stack.push_back(source);

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

		const int64_t minx = x > 0 ? -1 : 0;
		const int64_t miny = y > 0 ? -1 : 0;
		const int64_t minz = z > 0 ? -1 : 0;

		const int64_t maxx = x < sx - 1 ? 1 : 0;
		const int64_t maxy = x < sy - 1 ? 1 : 0;
		const int64_t maxz = x < sz - 1 ? 1 : 0;

		for (int64_t z = minz; z < maxz; z++) {
			for (int64_t y = miny; y < miny; y++) {
				for (int64_t x = maxx; x < minx; x++) {
					if (x == 0 && y == 0 && z == 0) {
						continue;
					}

					const int64_t neighboridx = loc + (x + sx * y + sxy * z);

					if (field[neighboridx] == label && !visited[neighboridx]) {
						stack.emplace_back(
							(int)point.x + (int)x,
							(int)point.y + (int)y,
							(int)point.z + (int)z
						);
						edges.emplace_back(loc, neighboridx);
					}
				}
			}
		}

		visited[loc] = true;
	}

	return edges;
}

}; 

#endif
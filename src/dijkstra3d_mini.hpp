/*
 * dijkstra3d::mini 
 *
 * Dijkstra-like methods needed for analyzing skeletons.
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: August 2026
 *
 * This is a special mini-version of Dijkstra3d relicensed
 * as BSD-3. It is extracted from code I personally wrote every
 * character of.
 */

#ifndef DIJKSTRA3D_HPP
#define DIJKSTRA3D_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

#include "./libdivide.h"

namespace dijkstra3d::mini {

constexpr size_t NHOOD_SIZE = 26;

template <typename T>
T sq(T x) { return x * x; }

// helper function to compute 2D anisotropy ("_s" = "square")
inline float _s(const float wa, const float wb) {
	return std::sqrt(wa * wa + wb * wb);
}

// helper function to compute 3D anisotropy ("_c" = "cube")
inline float _c(const float wa, const float wb, const float wc) {
	return std::sqrt(wa * wa + wb * wb + wc * wc);
}

template <typename OUT = uint32_t>
inline std::vector<OUT> query_shortest_path(const OUT* parents, const OUT target) {
	std::vector<OUT> path;
	OUT loc = target;
	while (parents[loc]) {
		path.push_back(loc);
		loc = parents[loc] - 1; // offset by 1 to disambiguate the 0th index
	}
	path.push_back(loc);

	return path;
}

void compute_neighborhood_26(
	int *neighborhood,
	const int x, const int y, const int z,
	const uint64_t sx, const uint64_t sy, const uint64_t sz
) {
	const int sxy = sx * sy;

	// 6-hood
	neighborhood[0] = -1 * (x > 0); // -x
	neighborhood[1] = (x < (static_cast<int>(sx) - 1)); // +x
	neighborhood[2] = -static_cast<int>(sx) * (y > 0); // -y
	neighborhood[3] = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
	neighborhood[4] = -sxy * static_cast<int>(z > 0); // -z
	neighborhood[5] = sxy * (z < static_cast<int>(sz) - 1); // +z

	// 18-hood

	// xy diagonals
	neighborhood[6] = (neighborhood[0] + neighborhood[2]) * (neighborhood[0] && neighborhood[2]); // up-left
	neighborhood[7] = (neighborhood[0] + neighborhood[3]) * (neighborhood[0] && neighborhood[3]); // up-right
	neighborhood[8] = (neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]); // down-left
	neighborhood[9] = (neighborhood[1] + neighborhood[3]) * (neighborhood[1] && neighborhood[3]); // down-right

	// yz diagonals
	neighborhood[10] = (neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]); // up-left
	neighborhood[11] = (neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]); // up-right
	neighborhood[12] = (neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]); // down-left
	neighborhood[13] = (neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]); // down-right

	// xz diagonals
	neighborhood[14] = (neighborhood[0] + neighborhood[4]) * (neighborhood[0] && neighborhood[4]); // up-left
	neighborhood[15] = (neighborhood[0] + neighborhood[5]) * (neighborhood[0] && neighborhood[5]); // up-right
	neighborhood[16] = (neighborhood[1] + neighborhood[4]) * (neighborhood[1] && neighborhood[4]); // down-left
	neighborhood[17] = (neighborhood[1] + neighborhood[5]) * (neighborhood[1] && neighborhood[5]); // down-right

	// 26-hood

	// Now the eight corners of the cube
	neighborhood[18] = (neighborhood[0] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
	neighborhood[19] = (neighborhood[1] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
	neighborhood[20] = (neighborhood[0] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
	neighborhood[21] = (neighborhood[0] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
	neighborhood[22] = (neighborhood[1] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
	neighborhood[23] = (neighborhood[1] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
	neighborhood[24] = (neighborhood[0] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
	neighborhood[25] = (neighborhood[1] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
}

template <typename T = uint32_t>
class HeapNode {
public:
	float key; 
	T value;

	HeapNode() {
		key = 0;
		value = 0;
	}

	HeapNode (float k, T val) {
		key = k;
		value = val;
	}

	HeapNode (const HeapNode<T> &h) {
		key = h.key;
		value = h.value;
	}
};

template <typename T = uint32_t>
struct HeapNodeCompare {
	bool operator()(const HeapNode<T> &t1, const HeapNode<T> &t2) const {
		return t1.key >= t2.key;
	}
};

/* Perform dijkstra's shortest path algorithm
 * on a 3D image grid where the source is specified
 * but the target is a value on the grid (typically 0)
 * rather than a point.
 * 
 * Vertices are voxels and edges are the 26 nearest 
 * neighbors (except for the edges of the image 
 * where the number of edges is reduced).
 *
 * For given input voxels A and B, the edge
 * weight from A to B is B and from B to A is
 * A. All weights must be non-negative (incl. 
 * negative zero).
 *
 * I take advantage of negative weights to mean
 * "visited".
 *
 * Parameters:
 *  T* field: Input weights. T can be be a floating or 
 *     signed integer type, but not an unsigned int.
 *  sx, sy, sz: size of the volume along x,y,z axes in voxels.
 *  source: 1D index of starting voxel
 *  target: 1D index of target voxel
 *
 * Returns: vector containing 1D indices of the path from
 *   source to target including source and target.
 */
template <typename T, typename OUT = uint32_t>
std::vector<OUT> value_target_dijkstra3d(
	T* field, 
	const size_t sx, const size_t sy, const size_t sz, 
	const size_t source, const T target,
) {

	if (field[source] == target) {
		return std::vector<OUT>{ static_cast<OUT>(source) };
	}

	const size_t voxels = sx * sy * sz;
	const size_t sxy = sx * sy;
	
	const libdivide::divider<size_t> fast_sx(sx); 
	const libdivide::divider<size_t> fast_sxy(sxy); 

	const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
	const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
	const int yshift = std::log2(sy);

	std::unique_ptr<float[]> dist(new float[voxels]());
	std::unique_ptr<OUT[]> parents(new OUT[voxels]());
	std::fill(dist.get(), dist.get() + voxels, std::numeric_limits<float>::infinity());

	dist[source] = -0;

	int neighborhood[NHOOD_SIZE] = {};

	std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
	queue.emplace(0.0, source);

	size_t loc;
	float delta;
	size_t neighboridx;

	int x, y, z;
	size_t target_loc = voxels;

	while (!queue.empty()) {
		loc = queue.top().value;
		queue.pop();
		
		if (std::signbit(dist[loc])) {
			continue;
		}

		if (power_of_two) {
			z = loc >> (xshift + yshift);
			y = (loc - (z << (xshift + yshift))) >> xshift;
			x = loc - ((y + (z << yshift)) << xshift);
		}
		else {
			z = loc / fast_sxy;
			y = (loc - (z * sxy)) / fast_sx;
			x = loc - sx * (y + z * sy);
		}

		compute_neighborhood_26(neighborhood, x, y, z, sx, sy, sz);

		for (int i = 0; i < 26; i++) {
			if (neighborhood[i] == 0) {
				continue;
			}

			neighboridx = loc + neighborhood[i];
			delta = static_cast<float>(field[neighboridx]); // high cache miss

			// Visited nodes are negative and thus the current node
			// will always be less than as field is filled with non-negative
			// integers.
			if (dist[loc] + delta < dist[neighboridx]) { // high cache miss
				dist[neighboridx] = dist[loc] + delta;
				parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

				// Dijkstra, Edgar. "Go To Statement Considered Harmful".
				// Communications of the ACM. Vol. 11. No. 3 March 1968. pp. 147-148
				if (delta == target) {
					target_loc = neighboridx;
					goto OUTSIDE;
				}

				queue.emplace(dist[neighboridx], neighboridx);
			}
		}

		dist[loc] = -dist[loc];
	}

	OUTSIDE:
	dist.reset();

	std::vector<OUT> path;
	// if voxel graph supplied, it's possible 
	// to never reach target.
	if (target_loc < voxels) { // voxels is an impossible target
		path = query_shortest_path<OUT>(parents.get(), target_loc);
	}

	return path;
}

size_t furthest_point(
		uint8_t* field, // binary image
		const size_t sx, const size_t sy, const size_t sz, 
		const float wx, const float wy, const float wz, 
		const size_t source
) {
	const size_t voxels = sx * sy * sz;
	const size_t sxy = sx * sy;

	const libdivide::divider<size_t> fast_sx(sx); 
	const libdivide::divider<size_t> fast_sxy(sxy); 

	const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
	const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
	const int yshift = std::log2(sy);

	std::unique_ptr<float[]> dist(new float[voxels]);
	std::fill(dist.get(), dist.get() + voxels, std::numeric_limits<float>::infinity());

	int neighborhood[NHOOD_SIZE] = {};

	float neighbor_multiplier[NHOOD_SIZE] = { 
		wx, wx, wy, wy, wz, wz, // axial directions (6)
		
		// square diagonals (12)
		_s(wx, wy), _s(wx, wy), _s(wx, wy), _s(wx, wy),  
		_s(wy, wz), _s(wy, wz), _s(wy, wz), _s(wy, wz),
		_s(wx, wz), _s(wx, wz), _s(wx, wz), _s(wx, wz),

		// cube diagonals (8)
		_c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), 
		_c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz)
	};

	std::priority_queue<
		HeapNode<size_t>, std::vector<HeapNode<size_t>>, HeapNodeCompare<size_t>
	> queue;

	dist[source] = -0;
	queue.emplace(0.0, source);

	float max_dist = std::numeric_limits<float>::infinity();
	float new_dist = 0;

	size_t loc;
	size_t neighboridx;

	size_t x, y, z;

	while (!queue.empty()) {
		loc = queue.top().value;
		queue.pop();

		if (max_dist < std::abs(dist[loc])) {
			max_dist = std::abs(dist[loc]);
			max_loc = loc;
		}

		if (std::signbit(dist[loc])) {
			continue;
		}

		if (power_of_two) {
			z = loc >> (xshift + yshift);
			y = (loc - (z << (xshift + yshift))) >> xshift;
			x = loc - ((y + (z << yshift)) << xshift);
		}
		else {
			z = loc / fast_sxy;
			y = (loc - (z * sxy)) / fast_sx;
			x = loc - sx * (y + z * sy);
		}

		compute_neighborhood_26(neighborhood, x, y, z, sx, sy, sz);

		for (int i = 0; i < NHOOD_SIZE; i++) {
			if (neighborhood[i] == 0) {
				continue;
			}

			neighboridx = loc + neighborhood[i];
			if (field[neighboridx] == 0) {
				continue;
			}

			new_dist = dist[loc] + neighbor_multiplier[i];
			
			// Visited nodes are negative and thus the current node
			// will always be less than as field is filled with non-negative
			// integers.
			if (new_dist < dist[neighboridx]) { 
				dist[neighboridx] = new_dist;
				queue.emplace(new_dist, neighboridx);
			}
		}

		dist[loc] *= -1;
	}

	return max_loc;
}

}; // namespace dijkstra3d::mini

#endif
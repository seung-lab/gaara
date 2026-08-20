#ifndef __GAARA_BINARY_HPP__
#define __GAARA_BINARY_HPP__

#include <cstdio>
#include <cstdint>
#include <deque>
#include <list>
#include <functional>
#include <stdexcept>

#include "builtins.hpp"
#include "def.hpp"
#include "postprocess.hpp"

using namespace gaara::def;

namespace gaara::binary {

// map for i variable (skip center to save space)

// z = -1
// 0 1 2 ; x axis
// 3 4 5
// 6 7 8

// z = 0
//  9 10 11 ; x axis
// 12 p  13
// 14 15 16

// z = +1
// 17 18 19 ; x axis
// 20 21 22
// 23 24 25

// y
// axis

template <typename LABEL, bool Interior>
uint32_t foreground_configuration(
	LABEL* labels, 
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const uint64_t x, const uint64_t y, const uint64_t z  
) {
	uint32_t index = 0;
	const uint64_t sxy = sx * sy;
	const uint64_t loc = x + sx * (y + sy * z);

    if constexpr (Interior) {
		// z = -1
		index |= (labels[loc-1-sx-sxy] > 0);
		index |= (labels[loc-sx-sxy] > 0) << 1;
		index |= (labels[loc+1-sx-sxy] > 0) << 2;
		
		index |= (labels[loc-1-sxy] > 0) << 3;
		index |= (labels[loc-sxy] > 0) << 4;
		index |= (labels[loc+1-sxy] > 0) << 5;

		index |= (labels[loc-1+sx-sxy] > 0) << 6;
		index |= (labels[loc+sx-sxy] > 0) << 7;
		index |= (labels[loc+1+sx-sxy] > 0) << 8;
		
		// z = 0
		index |= (labels[loc-1-sx] > 0) << 9;
		index |= (labels[loc-sx] > 0) << 10;
		index |= (labels[loc+1-sx] > 0) << 11;

		index |= (labels[loc-1] > 0) << 12;
		index |= (labels[loc+1] > 0) << 13;

		index |= (labels[loc-1+sx] > 0) << 14;
		index |= (labels[loc+sx] > 0) << 15;
		index |= (labels[loc+1+sx] > 0) << 16;

		// z = +1
		index |= (labels[loc-1-sx+sxy] > 0) << 17;
		index |= (labels[loc-sx+sxy] > 0) << 18;
		index |= (labels[loc+1-sx+sxy] > 0) << 19;

		index |= (labels[loc-1+sxy] > 0) << 20;
		index |= (labels[loc+sxy] > 0) << 21;
		index |= (labels[loc+1+sxy] > 0) << 22;

		index |= (labels[loc-1+sx+sxy] > 0) << 23;
		index |= (labels[loc+sx+sxy] > 0) << 24;
		index |= (labels[loc+1+sx+sxy] > 0) << 25;
    }
    else {
		const bool ltx = x < sx-1;
		const bool lty = y < sy-1;
		const bool ltz = z < sz-1;

		// z = -1
		index |= (x > 0) && (y > 0) && (z > 0) && (labels[loc-1-sx-sxy] > 0);
		index |= ((y > 0) && (z > 0) && (labels[loc-sx-sxy] > 0)) << 1;
		index |= (ltx && (y > 0) && (z > 0) && (labels[loc+1-sx-sxy] > 0)) << 2;
		
		index |= ((x > 0) && (z > 0) && (labels[loc-1-sxy] > 0)) << 3;
		index |= ((z > 0) && (labels[loc-sxy] > 0)) << 4;
		index |= (ltx && (z > 0) && (labels[loc+1-sxy] > 0)) << 5;

		index |= ((x > 0) && lty && (z > 0) && (labels[loc-1+sx-sxy] > 0)) << 6;
		index |= (lty && (z > 0) && (labels[loc+sx-sxy] > 0)) << 7;
		index |= (ltx && lty && (z > 0) && (labels[loc+1+sx-sxy] > 0)) << 8;
		
		// z = 0
		index |= ((x > 0) && (y > 0) && (labels[loc-1-sx] > 0)) << 9;
		index |= ((y > 0) && (labels[loc-sx] > 0)) << 10;
		index |= (ltx && (y > 0) && (labels[loc+1-sx] > 0)) << 11;

		index |= ((x > 0) && (labels[loc-1] > 0)) << 12;
		index |= (ltx && (labels[loc+1] > 0)) << 13;

		index |= ((x > 0) && lty && (labels[loc-1+sx] > 0)) << 14;
		index |= (lty && (labels[loc+sx] > 0)) << 15;
		index |= (ltx && lty && (labels[loc+1+sx] > 0)) << 16;

		// z = +1
		index |= ((x > 0) && (y > 0) && ltz && (labels[loc-1-sx+sxy] > 0)) << 17;
		index |= ((y > 0) && ltz && (labels[loc-sx+sxy] > 0)) << 18;
		index |= (ltx && (y > 0) && ltz && (labels[loc+1-sx+sxy] > 0)) << 19;

		index |= ((x > 0) && ltz && (labels[loc-1+sxy] > 0)) << 20;
		index |= (ltz && (labels[loc+sxy] > 0)) << 21;
		index |= (ltx && ltz && (labels[loc+1+sxy] > 0)) << 22;

		index |= ((x > 0) && lty && ltz && (labels[loc-1+sx+sxy] > 0)) << 23;
		index |= (lty && ltz && (labels[loc+sx+sxy] > 0)) << 24;
		index |= (ltx && lty && ltz && (labels[loc+1+sx+sxy] > 0)) << 25;
	}

	return index;
}

template <typename LABEL>
auto find_border_points(
	LABEL* labels,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool erode_border = true
) {

	// assume a 3x3x3 stencil with all voxels on
	const uint64_t sxy = sx * sy;

	auto is_pure = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi
	) {
		const uint64_t loc = xi + sx * (yi + sy * zi);

		if (erode_border) {
			return static_cast<bool>(
				(xi > 0 && xi < sx - 1 && yi > 0 && yi < sy - 1 && zi > 0 && zi < sz - 1)
				&& (labels[loc] != PointStatus::BACKGROUND)
				&& (labels[loc-sx] != PointStatus::BACKGROUND)
				&& (labels[loc+sx] != PointStatus::BACKGROUND)
				&& (labels[loc-sxy] != PointStatus::BACKGROUND)
				&& (labels[loc+sxy] != PointStatus::BACKGROUND)
				&& (labels[loc-sx-sxy] != PointStatus::BACKGROUND)
				&& (labels[loc+sx-sxy] != PointStatus::BACKGROUND)
				&& (labels[loc-sx+sxy] != PointStatus::BACKGROUND)
				&& (labels[loc+sx+sxy] != PointStatus::BACKGROUND)
			);
		}
		else {
			return static_cast<bool>(
				(xi >= 0 && xi < sx)
				&& (labels[loc] != PointStatus::BACKGROUND)
				&& ((yi == 0) || (yi > 0 && labels[loc-sx] != PointStatus::BACKGROUND))
				&& ((yi >= sy - 1) || (yi < sy - 1 && labels[loc+sx] != PointStatus::BACKGROUND))
				&& ((zi == 0) || (zi > 0 && labels[loc-sxy] != PointStatus::BACKGROUND))
				&& ((zi >= sz - 1) || (zi < sz - 1 && labels[loc+sxy] != PointStatus::BACKGROUND))
				&& ((yi == 0 || zi == 0) || (yi > 0 && zi > 0 && labels[loc-sx-sxy] != PointStatus::BACKGROUND))
				&& ((yi >= sy - 1 || zi == 0) || (yi < sy -1 && zi > 0 && labels[loc+sx-sxy] != PointStatus::BACKGROUND))
				&& ((yi == 0 || zi >= sz - 1) || (yi > 0 && zi < sz - 1 && labels[loc-sx+sxy] != PointStatus::BACKGROUND))
				&& ((yi >= sy - 1 || zi >= sz - 1) || (yi < sy - 1 && zi < sz - 1 && labels[loc+sx+sxy] != PointStatus::BACKGROUND))
			);
		}
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		bool pure_left = true;
		bool pure_middle = true;
		bool pure_right = true;

		int stale_stencil = 3;

		std::list<Voxel> border_points;

		for (uint64_t z = zs; z < ze; z++) {
			for (uint64_t y = ys; y < ye; y++) {
				stale_stencil = 3;
				for (uint64_t x = xs; x < xe; x++) {
					uint64_t loc = x + sx * (y + sy * z);

					if (labels[loc] == PointStatus::BACKGROUND) {
						stale_stencil++;
						continue;
					}

					if (stale_stencil == 1) {
						pure_left = pure_middle;
						pure_middle = pure_right;
						pure_right = is_pure(x+1,y,z);
					}
					else if (stale_stencil >= 3) {
						pure_right = is_pure(x+1,y,z);
						pure_middle = is_pure(x,y,z);
						pure_left = is_pure(x-1,y,z);
					}
					else if (stale_stencil == 2) {
						pure_left = pure_right;
						pure_right = is_pure(x+1,y,z);
						pure_middle = is_pure(x,y,z);
					}
					
					if (!pure_right || !pure_middle || !pure_left) {
						labels[loc] = PointStatus::BORDER;
						border_points.emplace_back(x,y,z);
					}

					stale_stencil = 1;
				}
			}
		}
		return border_points;
	};

	return process_block(0, sx, 0, sy, 0, sz);
}


template <typename LABEL>
uint64_t thin(
	LABEL* labels,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool preserve_endpoints = false,
	const int64_t max_iterations = -1
) {
	if (labels == nullptr) {
		throw std::invalid_argument("Null pointer provided for data.");
	}
	else if (sx >= gaara::def::MAX_DIM || sy >= gaara::def::MAX_DIM || sz >= gaara::def::MAX_DIM) {
		throw std::invalid_argument("Image is larger than maximum supported dimensions.");
	}
	else if (sx == 0 || sy == 0 || sz == 0) {
		return 0;
	}

	// enforce binary image starting point
	const uint64_t voxels = sx * sy * sz;
	const uint64_t sxy = sx * sy;

	for (uint64_t i = 0; i < voxels; i++) {
		labels[i] = labels[i] > 0; // PointStatus::BACKGROUND or FOREGROUND
	}

	using Iterator = std::list<Voxel>::iterator;

	std::list<Voxel> border_points = find_border_points(labels, sx, sy, sz);
	std::deque<Iterator> potentially_deletable;

	using BorderCheckFn = std::function<bool(const Voxel&, uint64_t)>;

	BorderCheckFn direction_map[6] = {
		[&](const Voxel& pt, uint64_t loc) { // +x
			return (pt.x > 0) && labels[loc-1] == PointStatus::BACKGROUND;
		},
		[&](const Voxel& pt, uint64_t loc) { // +y
			return (pt.y > 0) && labels[loc-sx] == PointStatus::BACKGROUND;
		},
		[&](const Voxel& pt, uint64_t loc) { // +z
			return (pt.z > 0) && labels[loc-sxy] == PointStatus::BACKGROUND;
		},
		[&](const Voxel& pt, uint64_t loc) { // -x
			return (pt.x < sx - 1) && labels[loc+1] == PointStatus::BACKGROUND;
		},
		[&](const Voxel& pt, uint64_t loc) { // -y
			return (pt.y < sy - 1) && labels[loc+sx] == PointStatus::BACKGROUND;
		},
		[&](const Voxel& pt, uint64_t loc) { // -z
			return (pt.z < sz - 1) && labels[loc+sxy] == PointStatus::BACKGROUND;
		}
	};

	auto kernel = [&](ThinningDirection direction) {
		uint64_t number_of_deleted_points = 0;
		potentially_deletable.clear();

		BorderCheckFn border_check_fn = direction_map[direction];

		for (auto it = border_points.begin(); it != border_points.end();) {
			const Voxel pt = *it;
			const uint64_t loc = pt.x + sx * (pt.y + sy * pt.z);

			// Should this ever happen?
			if (labels[loc] != PointStatus::BORDER) {
				it = border_points.erase(it);
				continue;
			}

			const bool interior = (
				pt.x > 0 && pt.x < sx - 1 
			 && pt.y > 0 && pt.y < sy - 1
			 && pt.z > 0 && pt.z < sz - 1
			);

			const uint32_t config = interior
				? foreground_configuration<LABEL, true>(labels, sx, sy, sz, pt.x, pt.y, pt.z)
				: foreground_configuration<LABEL, false>(labels, sx, sy, sz, pt.x, pt.y, pt.z);

			const bool is_directionally_border = border_check_fn(pt, loc);

			// Palagyi's algorithm puts isthmus after simple, but they are 
			// disjoint sets, so only one or zero should ever fire. I put
			// isthmus first since it has a simpler condition and we can then
			// put the simple decision behind an if else statement to avoid
			// some calculation.
			if (preserve_endpoints && popcount(config) == 1) {
				labels[loc] = PointStatus::PRESERVE;
				it = border_points.erase(it);
				continue;				
			}
			else if (isthmus_lut[config]) {
				labels[loc] = PointStatus::PRESERVE;
				it = border_points.erase(it);
				continue;
			}
			else if (is_directionally_border && simple_lut[config]) {
				potentially_deletable.emplace_back(it);
			}

			it++;
		}

		// Phase 2
		for (Iterator& it : potentially_deletable) {
			const Voxel pt = *it;
			
			const bool interior = (
				pt.x > 0 && pt.x < sx - 1 
			 && pt.y > 0 && pt.y < sy - 1
			 && pt.z > 0 && pt.z < sz - 1
			);

			const uint32_t config = interior
				? foreground_configuration<LABEL, true>(labels, sx, sy, sz, pt.x, pt.y, pt.z)
				: foreground_configuration<LABEL, false>(labels, sx, sy, sz, pt.x, pt.y, pt.z);

			if (!simple_lut[config]) {
				continue;
			}

			const uint64_t loc = pt.x + sx * (pt.y + sy * pt.z);
			labels[loc] = PointStatus::BACKGROUND;
			border_points.erase(it);
			number_of_deleted_points++;

			if (pt.x > 0 && labels[loc-1] == PointStatus::FOREGROUND) {
				labels[loc-1] = PointStatus::BORDER;
				border_points.emplace_back(pt.x - 1, pt.y, pt.z);
			}
			if (pt.y > 0 && labels[loc-sx] == PointStatus::FOREGROUND) {
				labels[loc-sx] = PointStatus::BORDER;
				border_points.emplace_back(pt.x, pt.y - 1, pt.z);
			}
			if (pt.z > 0 && labels[loc-sxy] == PointStatus::FOREGROUND) {
				labels[loc-sxy] = PointStatus::BORDER;
				border_points.emplace_back(pt.x, pt.y, pt.z - 1);
			}
			if (pt.x < sx - 1 && labels[loc+1] == PointStatus::FOREGROUND) {
				labels[loc+1] = PointStatus::BORDER;
				border_points.emplace_back(pt.x + 1, pt.y, pt.z);
			}
			if (pt.y < sy - 1 && labels[loc+sx] == PointStatus::FOREGROUND) {
				labels[loc+sx] = PointStatus::BORDER;
				border_points.emplace_back(pt.x, pt.y + 1, pt.z);
			}
			if (pt.z < sz - 1 && labels[loc+sxy] == PointStatus::FOREGROUND) {
				labels[loc+sxy] = PointStatus::BORDER;
				border_points.emplace_back(pt.x, pt.y, pt.z + 1);
			}
		}
	
		return number_of_deleted_points;
	};

	uint64_t number_of_deleted_points = 0;
	uint64_t num_iterations = 0;
	do {
		number_of_deleted_points = 0;
		number_of_deleted_points += kernel(ThinningDirection::PLUS_X);
		number_of_deleted_points += kernel(ThinningDirection::MINUS_X);
		number_of_deleted_points += kernel(ThinningDirection::PLUS_Y);
		number_of_deleted_points += kernel(ThinningDirection::MINUS_Y);
		number_of_deleted_points += kernel(ThinningDirection::PLUS_Z);
		number_of_deleted_points += kernel(ThinningDirection::MINUS_Z);
		num_iterations++;
	} while (number_of_deleted_points > 0 && (max_iterations < 0 || num_iterations < max_iterations));

	for (uint64_t i = 0; i < voxels; i++) {
		labels[i] = labels[i] > 0;
	}

	return num_iterations;
}


template <typename LABEL>
auto skeletonize(
	LABEL* labels,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool preserve_endpoints = false
) {
	int64_t iters = thin(labels, sx, sy, sz, preserve_endpoints);
	return gaara::postprocess::extract_skeletons(labels, sx, sy, sz)[1];
}

};

#endif
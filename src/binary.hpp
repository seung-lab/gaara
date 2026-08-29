#ifndef __GAARA_BINARY_HPP__
#define __GAARA_BINARY_HPP__

#include <cstdio>
#include <cstdint>
#include <deque>
#include <forward_list>
#include <functional>
#include <stdexcept>
#include <thread>

#include "builtins.hpp"
#include "def.hpp"
#include "postprocess.hpp"
#include "threadpool.hpp"

using namespace gaara::def;

namespace gaara::binary {


using Iterator = std::forward_list<Voxel>::iterator;

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
	const bool erode_border,
	GaaraThreadPool& pool
) {

	const unsigned int threads = pool.num_threads();

	// assume a 3x3x3 stencil with all voxels on
	const uint64_t sxy = sx * sy;

	auto is_pure = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi
	) {
		const uint64_t loc = xi + sx * (yi + sy * zi);

		if (erode_border) {
			return static_cast<bool>(
				(xi >= 0 && xi < sx && yi > 0 && yi < sy - 1 && zi > 0 && zi < sz - 1)
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

	std::vector<std::forward_list<Voxel>> border_points(threads);

	auto process_block = [&](
		const unsigned int t,
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		bool pure_left = true;
		bool pure_middle = true;
		bool pure_right = true;

		int stale_stencil = 3;

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
						if (x < sx - 1) {
							pure_right = is_pure(x+1,y,z);
						}
						else {
							pure_right = !erode_border;
						}
					}
					else if (stale_stencil >= 3) {
						if (x < sx - 1) {
							pure_right = is_pure(x+1,y,z);
						}
						else {
							pure_right = !erode_border;
						}
						pure_middle = is_pure(x,y,z);
						if (x > 0) {
							pure_left = is_pure(x-1,y,z);
						}
						else {
							pure_left = !erode_border;
						}
					}
					else if (stale_stencil == 2) {
						pure_left = pure_right;
						if (x < sx - 1) {
							pure_right = is_pure(x+1,y,z);
							pure_middle = is_pure(x,y,z);
						}
						else {
							pure_middle = is_pure(x,y,z);
							pure_right = !erode_border;
						}
					}
					
					if (!pure_right || !pure_middle || !pure_left) {
						labels[loc] = PointStatus::BORDER;
						border_points[t].emplace_front(x,y,z);
					}

					stale_stencil = 1;
				}
			}
		}
	};

	uint64_t cz = (sz + threads - 1) / threads;
	cz = std::max(cz, (uint64_t)1);

	std::vector<std::function<void(std::size_t)>> jobs;
	jobs.reserve(threads);
	for (int t = 0; t < threads; t++) {
		jobs.emplace_back([&,t](size_t ignore) {
			const uint64_t start = t * cz;
			const uint64_t end = std::min((t+1) * cz, sz);
			process_block(t, 0, sx, 0, sy, start, end);
		});
	}

	pool.run_batch(jobs);
	
	return border_points;
}

template <typename LABEL>
auto find_border_points(
	LABEL* labels,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool erode_border = true,
	const unsigned int threads = 1
) {
	GaaraThreadPool pool(threads);
	return find_border_points(labels, sx, sy, sz, erode_border, pool);
}

template <typename LABEL>
uint64_t kernel(
	ThinningDirection direction,
	LABEL* labels, 
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	std::vector<std::forward_list<Voxel>>& border_points,
	std::vector<std::deque<Iterator>>& potentially_deletable,
	const bool preserve_endpoints,
	const unsigned int threads,
	GaaraThreadPool& pool
) {
	using BorderCheckFn = std::function<bool(const Voxel&, uint64_t)>;

	const uint64_t sxy = sx * sy;

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

	for (size_t t = 0; t < threads; t++) {
		potentially_deletable[t].clear();
	}

	BorderCheckFn border_check_fn = direction_map[direction];

	auto phase1 = [&](std::size_t t) {
		auto it = border_points[t].begin();
		auto prev = border_points[t].before_begin();

		while (it != border_points[t].end()) {
			const Voxel pt = *it;
			const uint64_t loc = pt.x + sx * (pt.y + sy * pt.z);

			// Should this ever happen?
			if (labels[loc] != PointStatus::BORDER) {
				it = border_points[t].erase_after(prev);
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

			// Palagyi's algorithm puts isthmus after simple, but they are 
			// disjoint sets, so only one or zero should ever fire. I put
			// isthmus first since it has a simpler condition and we can then
			// put the simple decision behind an if else statement to avoid
			// some calculation.
			if (preserve_endpoints && popcount(config) == 1) {
				labels[loc] = PointStatus::PRESERVE;
				it = border_points[t].erase_after(prev);
				continue;				
			}
			else if (isthmus_lut[config]) {
				labels[loc] = PointStatus::PRESERVE;
				it = border_points[t].erase_after(prev);
				continue;
			}
			else if (border_check_fn(pt, loc) && simple_lut[config]) {
				potentially_deletable[t].emplace_front(prev);
			}

			prev = it;
			it++;
		}
	};

	std::vector<std::function<void(std::size_t)>> jobs;
	jobs.reserve(threads);
	for (std::size_t t = 0; t < threads; t++) {
		jobs.emplace_back([&,t](std::size_t ignore) { phase1(t); });
	}
	pool.run_batch(jobs);

	std::vector<uint64_t> number_of_deleted_points(threads);

	uint64_t cz = (sz + threads - 1) / threads;
	cz = std::max(cz, (uint64_t)2); // cz must be >= 2 to avoid lock confliction

	std::mutex splice_guard;
	std::vector<std::mutex> boundary_deconfliction(threads);

	// Phase 2
	auto phase2 = [&](unsigned int t) {
		std::vector<std::forward_list<Voxel>> outgoing(threads);

		std::unique_lock<std::mutex> boundary_lock; 

		for (Iterator& prev : potentially_deletable[t]) {
			auto it = std::next(prev);
			if (it == border_points[t].end()) {
        		continue;
    		}
			const Voxel pt = *it;
			
			const bool interior = (
				pt.x > 0 && pt.x < sx - 1 
			 && pt.y > 0 && pt.y < sy - 1
			 && pt.z > 0 && pt.z < sz - 1
			);

			// CRITICAL: require cz > 1 or else deadlocks can happen
			if (pt.z > 0 && pt.z == cz * t) {
				boundary_lock = std::unique_lock<std::mutex>(boundary_deconfliction[t - 1]);
			}
			else if (pt.z < sz - 1 && pt.z == (cz * (t+1)) - 1) {
				boundary_lock = std::unique_lock<std::mutex>(boundary_deconfliction[t]);
			}

			const uint32_t config = interior
				? foreground_configuration<LABEL, true>(labels, sx, sy, sz, pt.x, pt.y, pt.z)
				: foreground_configuration<LABEL, false>(labels, sx, sy, sz, pt.x, pt.y, pt.z);

			if (!simple_lut[config]) {
				if (boundary_lock.owns_lock()) {
					boundary_lock.unlock();
				}
				continue;
			}

			const uint64_t loc = pt.x + sx * (pt.y + sy * pt.z);
			labels[loc] = PointStatus::BACKGROUND;
			border_points[t].erase_after(prev);
			number_of_deleted_points[t]++;

			if (pt.x > 0 && labels[loc-1] == PointStatus::FOREGROUND) {
				labels[loc-1] = PointStatus::BORDER;
				outgoing[t].emplace_front(pt.x - 1, pt.y, pt.z);
			}
			if (pt.y > 0 && labels[loc-sx] == PointStatus::FOREGROUND) {
				labels[loc-sx] = PointStatus::BORDER;
				outgoing[t].emplace_front(pt.x, pt.y - 1, pt.z);
			}
			if (pt.z > 0 && labels[loc-sxy] == PointStatus::FOREGROUND) {
				labels[loc-sxy] = PointStatus::BORDER;
    			int t_owner = (pt.z == t * cz) ? (t - 1) : t;
    			t_owner = std::max(t_owner, 0);
				outgoing[t_owner].emplace_front(pt.x, pt.y, pt.z - 1);
			}
			if (pt.x < sx - 1 && labels[loc+1] == PointStatus::FOREGROUND) {
				labels[loc+1] = PointStatus::BORDER;
				outgoing[t].emplace_front(pt.x + 1, pt.y, pt.z);
			}
			if (pt.y < sy - 1 && labels[loc+sx] == PointStatus::FOREGROUND) {
				labels[loc+sx] = PointStatus::BORDER;
				outgoing[t].emplace_front(pt.x, pt.y + 1, pt.z);
			}
			if (pt.z < sz - 1 && labels[loc+sxy] == PointStatus::FOREGROUND) {
				labels[loc+sxy] = PointStatus::BORDER;
    			int t_owner = (pt.z + 1 == (t + 1) * cz) ? (t + 1) : t;
    			t_owner = std::min(t_owner, ((int)threads)-1);
				outgoing[t_owner].emplace_front(pt.x, pt.y, pt.z + 1);
			}

			if (boundary_lock.owns_lock()) {
				boundary_lock.unlock();
			}
		}

		std::unique_lock<std::mutex> lock(splice_guard);
		for (int t = 0; t < outgoing.size(); t++) {
			border_points[t].splice_after(border_points[t].before_begin(), outgoing[t]);
		}	
	};

	// This logic isn't obvious and interacts subtly with find_border_points.
	// phase2 is difficult to parallelize because each voxel is pulling in
	// neighbors that could be changing. Previously, this algorithm used a 
	// round-robin scheduler that was easy to parallelize for phase1 but not
	// phase2.

	// Moving to z-slabs allowed superior parallelism in find_border_points
	// which exploited the sliding window mechanism better. It also clustered
	// border points per a slab, whose interiors are robust to conflict. However,
	// the neighboring slabs could confict if they were running concurrently. 
	// So we do even slabs then odd slabs. This allows each slab to peek into
	// its neighbor's territory, but halves the parallelism.

	// Second note: The high kernel usage is due to uneven distribution of border
	// points per a slab. Some threads finish early and keep checking the pool.
	// These issues are fixable, but require some more effort.

	jobs.clear();
	for (std::size_t t = 0; t < threads; t++) {
		jobs.emplace_back([&,t](std::size_t ignore) { phase2(t); });
	}
	pool.run_batch(jobs);

	uint64_t number_of_deleted_points_total = 0;
	for (unsigned int t = 0; t < threads; t++) {
		number_of_deleted_points_total += number_of_deleted_points[t];
	}

	return number_of_deleted_points_total;
}

template <typename LABEL>
void mask(LABEL* labels, const uint64_t voxels, GaaraThreadPool& pool) {
	if (voxels == 0) {
		return;
	}

	uint64_t chunk_size = voxels;
	unsigned int threads = pool.num_threads();
	const uint64_t min_chunk = 1000;

	constexpr uint64_t one = 1;

	if (min_chunk * threads > voxels) {
		threads = std::max(voxels / min_chunk, one);
		chunk_size = min_chunk;
	}
	else {
		chunk_size = (voxels + threads - 1) / threads;
		chunk_size = std::max(chunk_size, min_chunk);
	}
	
	std::vector<std::function<void(std::size_t)>> jobs;
	jobs.reserve(threads);
	for (unsigned int t = 0; t < threads; t++) {
		jobs.emplace_back([&,t](std::size_t ignore) {
			uint64_t end = std::min(chunk_size * (t+1), voxels);
			for (uint64_t i = chunk_size * t; i < end; i++) {
				labels[i] = labels[i] > 0; // PointStatus::BACKGROUND or FOREGROUND
			}
		});
	}
	pool.run_batch(jobs);
}

template <typename LABEL>
uint64_t thin(
	LABEL* labels,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool preserve_endpoints = false,
	const std::vector<Voxel> anchors = {}, 
	const int64_t max_iterations = -1,
	unsigned int threads = 1
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

	if (threads <= 0) {
		threads = std::thread::hardware_concurrency();
	}
	threads = std::min(threads, std::thread::hardware_concurrency());

	// cz must be >= 2 to avoid deadlocks in phase 2
	// work lists are assigned based on thread count.
	if (sz / threads == 0) {
		threads = std::max((unsigned int)(sz+1) / 2, (unsigned int)1);
	}

	// enforce binary image starting point
	const uint64_t voxels = sx * sy * sz;

	GaaraThreadPool pool(threads);

	mask(labels, voxels, pool);

	std::vector<std::forward_list<Voxel>> border_points = find_border_points(
		labels, sx, sy, sz, /*erode_border=*/true, pool
	);
	std::vector<std::deque<Iterator>> potentially_deletable(threads);

	uint64_t number_of_deleted_points = 0;
	int64_t num_iterations = 0;

	for (auto& pt : anchors) {
		const uint64_t loc = pt.x + sx * (pt.y + sy * pt.z);
		if (loc >= voxels) {
			throw std::invalid_argument("anchors must be located inside the image.");
		}

		if (labels[loc] != PointStatus::BACKGROUND) {
			labels[loc] = PointStatus::PRESERVE;
		}
	}

	// U,N,E,S,W,D

	do {
		number_of_deleted_points = 0;
		number_of_deleted_points += kernel(ThinningDirection::PLUS_Y, labels, sx, sy, sz, border_points, potentially_deletable, preserve_endpoints, threads, pool);
		number_of_deleted_points += kernel(ThinningDirection::PLUS_Z, labels, sx, sy, sz, border_points, potentially_deletable, preserve_endpoints, threads, pool);
		number_of_deleted_points += kernel(ThinningDirection::PLUS_X, labels, sx, sy, sz, border_points, potentially_deletable, preserve_endpoints, threads, pool);
		number_of_deleted_points += kernel(ThinningDirection::MINUS_Z, labels, sx, sy, sz, border_points, potentially_deletable, preserve_endpoints, threads, pool);
		number_of_deleted_points += kernel(ThinningDirection::MINUS_X, labels, sx, sy, sz, border_points, potentially_deletable, preserve_endpoints, threads, pool);
		number_of_deleted_points += kernel(ThinningDirection::MINUS_Y, labels, sx, sy, sz, border_points, potentially_deletable, preserve_endpoints, threads, pool);		
		num_iterations++;
	} while (number_of_deleted_points > 0 && (max_iterations < 0 || num_iterations < max_iterations));

	mask(labels, voxels, pool);

	pool.join();

	return number_of_deleted_points;
}


template <typename LABEL>
auto skeletonize(
	LABEL* labels,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool preserve_endpoints = false,
	const std::vector<Voxel> anchors = {},
	unsigned int threads = 1
) {
	thin(labels, sx, sy, sz, preserve_endpoints, anchors, -1, threads);
	return gaara::postprocess::extract_skeletons(labels, sx, sy, sz)[1];
}

};

#endif

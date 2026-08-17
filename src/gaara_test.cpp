// brew install google-test
// clang++ -Og -g src/gaara_test.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lgtest -lgtest_main -std=c++17 -o automated_tests && ./automated_tests

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <cstdio>

#include "gaara.hpp"

TEST(Gaara, TestFindBorderPoints) {

	int sz = 99;
	int sy = 101;
	int sx = 105;
	int voxels = sx * sy * sz;

	std::vector<uint8_t> image(voxels);
	for (uint64_t z = 1; z < sz-1; z++) {
		for (uint64_t y = 1; y < sy-1; y++) {
			for (uint64_t x = 1; x < sx-1; x++) {
				uint64_t loc = x + sx * (y + sy * z);
				image[loc] = 1;
			}
		}
	}

	auto border_points = gaara::find_border_points(image.data(), sx, sy, sz);

	// For debugging:

	// for (auto vx : border_points) {
	// 	printf("%d %d %d\n", vx.x, vx.y, vx.z);
	// }

	EXPECT_EQ(border_points.size(), (sx-2) * (sy-2) * (sz-2) - (sx-4) * (sy-4) * (sz-4));


}
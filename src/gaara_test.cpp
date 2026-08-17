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
	
	std::fill(image.begin(), image.end(), 1);
	border_points = gaara::find_border_points(image.data(), sx, sy, sz);

	EXPECT_EQ(border_points.size(), sx * sy * sz - (sx-2) * (sy-2) * (sz-2));

	border_points = gaara::find_border_points(image.data(), sx, sy, sz, false);

	EXPECT_EQ(border_points.size(), 0);
}

TEST(Gaara, TestSimplePointLUT) {

	EXPECT_EQ(gaara::simple_lut[0] , false);
	EXPECT_THROW(gaara::simple_lut[0xfffffffff] , std::runtime_error); // > num entries
	EXPECT_EQ(gaara::simple_lut[0b11111111111111111111111111], false); // 2^26 - 1
	EXPECT_EQ(gaara::simple_lut[0b00000000000000000000010000], false); // 4 (endpoint)
	EXPECT_EQ(gaara::simple_lut[0b00000000000000010000000000], false); // 10 (endpoint)
	EXPECT_EQ(gaara::simple_lut[0b11111111111111101111101111], true); // 4 & 10
	EXPECT_EQ(gaara::simple_lut[0b11111111111110101111101111], true); // 4 & 10 & 12
	EXPECT_EQ(gaara::simple_lut[0b11111111111111111111101010], true); // 0 & 2 & 4
}




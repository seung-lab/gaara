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

	for (int i = 0; i < 26; i++) {
		EXPECT_EQ(gaara::simple_lut[1 << i], false); // endpoint tests
	}

	EXPECT_EQ(gaara::simple_lut[0b11111111111111101111101111], true); // 4 & 10 off
	EXPECT_EQ(gaara::simple_lut[0b11111111111110101111101111], true); // 4 & 10 & 12 off
	EXPECT_EQ(gaara::simple_lut[0b11111111111111111111101010], true); // 0 & 2 & 4 off

	EXPECT_EQ(gaara::simple_lut[0b00000000000000010000010000], true); // 4 & 10 (endpoint)

}

TEST(Gaara, TestTorusSmall) {

	int sz = 3;
	int sy = 5;
	int sx = 5;
	int voxels = sx * sy * sz;

	std::vector<uint8_t> torus = {
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 1, 1, 1, 0,
		0, 1, 0, 1, 0,
		0, 1, 1, 1, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0
	};

	gaara::thin_palagyi(torus.data(), sx, sy, sz);

	int num_foreground = 0;
	for (int i = 0; i < voxels; i++) {
		num_foreground += torus[i] > 0;
	}

	EXPECT_EQ(num_foreground, 4);

	gaara::thin_palagyi(torus.data(), sx, sy, sz);

	num_foreground = 0;
	for (int i = 0; i < voxels; i++) {
		num_foreground += torus[i] > 0;
	}

	EXPECT_EQ(num_foreground, 4); // idempotent at this point
}

TEST(Gaara, TestTorusSmallDoubleThick) {
	int sz = 3;
	int sy = 7;
	int sx = 7;
	int voxels = sx * sy * sz;

	std::vector<uint8_t> torus = {
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,

		0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 0, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 0,
		0, 0, 0, 0, 0, 0, 0,

		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0
	};

	gaara::thin_palagyi(torus.data(), sx, sy, sz);

	// for (uint64_t y = 0; y < sy; y++) {
	// 	for (uint64_t x = 0; x < sx; x++) {
	// 		printf("%d ", torus[x + sx * y + sx * sy * 1]);
	// 	}
	// 	printf("\n");
	// }


	int num_foreground = 0;
	for (int i = 0; i < voxels; i++) {
		num_foreground += torus[i] > 0;
	}

	EXPECT_EQ(num_foreground, 7);

	gaara::thin_palagyi(torus.data(), sx, sy, sz);

	num_foreground = 0;
	for (int i = 0; i < voxels; i++) {
		num_foreground += torus[i] > 0;
	}

	EXPECT_EQ(num_foreground, 7); // idempotent at this point
}



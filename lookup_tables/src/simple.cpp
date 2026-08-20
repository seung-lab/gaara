// Palágyi, referencing Malandain and Bertrand noted that
// a simple point (p) in the center of a 3x3x3 binary space containing 
// foreground (1) and background (0) must satisfy:
//
// 1. The 26-connected foreground set without p must have 1 connected component.
// 2. The 6-connected neighborhood around p must have at least 1 background voxel.
// 3. At least two background points the 6-connected neighborhood must touch on their
//    edges (be 18-connected).
//
// Endpoints are considered "not simple" regardless of the above definition.
// An endpoint consists of a 3x3x3 stencil around point p that contains exactly
// one foreground voxel.
//
// This will generate 2^26 different permutations, which can be bit packed
// into 8,388,608 bytes which can be saved into a compressible data file.

// References
// 
// 1. K. Palágyi et al., “A Sequential 3D Thinning Algorithm and Its Medical Applications,”
//    in Information Processing in Medical Imaging, M. F. Insana and R. M. Leahy, 
//    Eds., Berlin, Heidelberg: Springer, 2001, pp. 409–415.
//    doi: 10.1007/3-540-45729-1_42.
// 2. Malandain, G., Bertrand, G.: Fast characterization of 3D simple points. 
//    In: Proc.  11th IEEE Internat. Conf. on Pattern Recognition, ICPR 1992, 
//    pp. 232–235 (1992)

// William Silversmith, August 2026
// Princeton Neuroscience Institute
// BSD-3 License

#include <cstdio>
#include <cstdint>

#include "cc3d.hpp"

// map for i variable (skip center to save space)

// z = 0
// 0 1 2 ; x axis
// 3 4 5
// 6 7 8

// z = 1
//  9 10 11 ; x axis
// 12 p  13
// 14 15 16

// z = 2
// 17 18 19 ; x axis
// 20 21 22
// 23 24 25

// y
// axis

const ssize_t SIZE = 1UL << 26;
const ssize_t PACKED_SIZE = (SIZE + 7) / 8;

uint8_t* paint(uint32_t i) {
	static uint8_t stencil[27];
	stencil[13] = 0;

	for (int j = 0; j < 27; j++) {
		int k = j;
		if (k == 13) {
			continue;
		}
		if (k > 13) {
			k--;
		}

		stencil[j] = ((i >> k) & 0b1); 
	}

	return stencil;
}

void print_stencil(uint8_t stencil[27]) {
	for (int z = 0; z < 3; z++) {
		printf("z=%d\n", z);
		for (int y = 0; y < 3; y++) {
			for (int x = 0; x < 3; x++) {
				printf("%d,", stencil[x + 3 * y + 9 * z]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

uint8_t* compute_oracle() {
	uint8_t* oracle = new uint8_t[SIZE]();
	uint8_t* out_labels = new uint8_t[27]();

	std::size_t N = 0;

	// generate every combination
	for (uint32_t i = 0; i < SIZE; i++) {
		uint8_t* stencil = paint(i);

		if (stencil[13] != 0) {
			printf("CRITICAL ERROR\n");
			exit(1);
		}

		// printf("i=%d\n", i);
		// print_stencil(stencil);

		N = 0;
		cc3d::connected_components3d_26<uint8_t, uint8_t>(
			stencil,
			/*sx=*/3, /*sy=*/3, /*sz=*/3, 
			/*max_labels=*/27,
			/*out_labels=*/out_labels,
			/*N=*/N
		);

		// N*_26(p) has 1 connected component
		const bool condition_1 = N == 1;

		// z = 0
		// 0 1 2 ; x axis
		// 3 4 5
		// 6 7 8

		// z = 1
		//  9 10 11 ; x axis
		// 12 13 14
		// 15 16 17

		// z = 2
		// 18 19 20 ; x axis
		// 21 22 23
		// 24 25 26

		const int num_6_bg = (
			(stencil[4] == 0)
			+ (stencil[10] == 0)
			+ (stencil[12] == 0)
			+ (stencil[14] == 0)
			+ (stencil[16] == 0)
			+ (stencil[22] == 0)
		);

		// N_6(p)\B has at least one bg point
		const bool condition_2 = num_6_bg > 0;		

		// N_6(p)\B has at least two points 18-connected
		const bool condition_3 = (
			(num_6_bg == 1)
			|| (
				((stencil[4] == 0) || (stencil[22] == 0)) 
				&& (
					   (stencil[10] == 0)
					|| (stencil[12] == 0)
					|| (stencil[14] == 0)
					|| (stencil[16] == 0)
				)
			)
			|| (
				((stencil[12] == 0) || (stencil[14] == 0))
				&& (
					   (stencil[4] == 0)
					|| (stencil[10] == 0)
					|| (stencil[16] == 0)
					|| (stencil[22] == 0)
				)
			)
			|| (
				((stencil[10] == 0) || (stencil[16] == 0))
				&& (
					   (stencil[4] == 0)
					|| (stencil[12] == 0)
					|| (stencil[14] == 0)
					|| (stencil[22] == 0)
				)
			)
		);

		int num_foreground = 0;
		for (int j = 0; j < 27; j++) {
			num_foreground += stencil[j] > 0;
		}

		const bool is_endpoint = (num_foreground == 1);

		oracle[i] = (uint8_t)(!is_endpoint && condition_1 && condition_2 && condition_3);
	}

	delete[] out_labels;

	return oracle;
}

void write_packed_oracle(uint8_t* oracle) {
	const char* filename = "tables/simple.bin";

	FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening file");
        return;
    }

    size_t length = fwrite(oracle, sizeof(oracle[0]), PACKED_SIZE, file);

    printf("wrote %s: %zu bytes\n", filename, length);

    fclose(file);
}

uint8_t* bit_pack_file(uint8_t* oracle) {
	uint8_t* packed_oracle = new uint8_t[PACKED_SIZE]();

	for (int i = 0, pi = 0; i < SIZE; i += 8, pi++) {
		uint8_t packed_word = 0;
		for (int j = 0; j < 8; j++) {
			packed_word |= (oracle[i+j] << j);
		}
		packed_oracle[pi] = packed_word;
	}

	return packed_oracle;
}

int main() {
	uint8_t* oracle = compute_oracle();
	uint8_t* packed_oracle = bit_pack_file(oracle);
	delete[] oracle;
	write_packed_oracle(packed_oracle);
	delete[] packed_oracle;

	return 0;
}
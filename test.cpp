#include "crackle.hpp"

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <iostream>
#include <chrono>

#include "src/binary.hpp"
#include "src/multilabel.hpp"

// to compile:
// clang++ -g -O3 -std=c++20 -Isrc -Ilibcrackle/ test.cpp -o test

std::string slurp(const std::string& filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary | std::ios::ate);
    if (!in) throw std::ios_base::failure("Failed to open file");

    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    std::string contents;
    contents.resize(size);
    in.read(&contents[0], size);
    return contents;
}

int main () {

	uint32_t soma = 25024949;
	uint32_t neurite = 71260610;
	uint32_t glia = 28336523;

	std::string binary = slurp("connectomics_filled.ckl");
	uint8_t* image = crackle::decompress<uint32_t, uint8_t>(binary, NULL, -1, -1, -1, soma);
	crackle::CrackleHeader header(binary);

	auto start = std::chrono::high_resolution_clock::now();
	gaara::binary::thin<uint8_t>(
		image,
		header.sx, header.sy, header.sz,
		false, {}, -1,
		/*threads=*/3
	);	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken: " << duration.count() << " us\n";

	delete[] image;

	return 0;
}




#ifndef __GAARA_DEFINITIONS_HPP__
#define __GAARA_DEFINITIONS_HPP__

#include <concepts>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef GAARA_STATICALLY_LINK_LUTS
	#include "simple_lut.hpp"
	#include "isthmus_lut.hpp"
#endif

namespace gaara::def {

constexpr uint64_t MAX_DIM{ std::numeric_limits<uint16_t>::max() };

enum PointStatus {
	BACKGROUND = 0,
	FOREGROUND = 1,
	BORDER = 2,
	PRESERVE = 3
};

struct OneBitArray {
	uint8_t* m_data;
	uint64_t m_size;
	uint64_t m_num_entries;
	bool m_static_data;
	const uint64_t k_max_size = (1 << 26) >> 3;

	OneBitArray(const uint64_t size) {
		m_num_entries = size;
		m_size = (size + 7) >> 3;
		m_data = new uint8_t[m_size]();
		m_static_data = false;
	}

	OneBitArray(const unsigned char* table, const unsigned int size) {
		m_data = const_cast<unsigned char*>(table);
		m_size = size;
		m_num_entries = size << 3;
		m_static_data = true;
	}

	OneBitArray(const char* filename) {
		std::ifstream file(filename, std::ios::binary | std::ios::ate);
		if (!file) {
			throw std::runtime_error("Failed to open file");
		}

		m_size = file.tellg();

		if (m_size < 0 || m_size != k_max_size) {
			throw std::runtime_error("Incorrect file size for lookup table.");
		}

		m_num_entries = m_size << 3;

		file.seekg(0, std::ios::beg);

		m_data = new uint8_t[m_size];
		if (!file.read(reinterpret_cast<char*>(m_data), m_size)) {
			throw std::runtime_error("Failed to read file data");
		}

		m_static_data = false;
	}

	~OneBitArray() {
		if (!m_static_data) {
			delete[] m_data;
		}
	}

	uint64_t size() const {
		return m_num_entries;
	}

	bool operator[](const uint64_t index) const {
		if (index >= m_num_entries) { 
			throw std::runtime_error("index greater than table size.");
		}
		const uint64_t offset = index >> 3;
		const uint64_t remainder = index & 0b111;
		return (m_data[offset] >> remainder) & 1;
	}

	void fill(const bool value) {
		const uint8_t packed_value = value
			? 0xff
			: 0x00;

		std::fill(m_data, m_data + m_size, packed_value);
	}
};

#ifdef GAARA_STATICALLY_LINK_LUTS 
	static OneBitArray simple_lut(gaara::lut::simple, gaara::lut::simple_len);
	static OneBitArray isthmus_lut(gaara::lut::isthmus, gaara::lut::isthmus_len);
#else
	static OneBitArray simple_lut("lookup_tables/tables/simple.bin");
	static OneBitArray isthmus_lut("lookup_tables/tables/isthmus.bin");
#endif

struct Voxel {
	uint16_t x;
	uint16_t y;
	uint16_t z;

	Voxel(uint16_t _x, uint16_t _y, uint16_t _z) : x(_x), y(_y), z(_z) {}
};

struct Skeleton {
	std::vector<uint64_t> vertices;
	std::vector<std::pair<uint64_t, uint64_t>> edges;

	Skeleton() {}

	Skeleton(std::vector<uint64_t>& _vertices, std::vector<std::pair<uint64_t, uint64_t>>& _edges)
		: vertices(_vertices), edges(_edges) {}
};

enum ThinningDirection {
	PLUS_X = 0,
	PLUS_Y = 1,
	PLUS_Z = 2,
	MINUS_X = 3,
	MINUS_Y = 4,
	MINUS_Z = 5
};

struct TwoBitArray {
	uint8_t* m_data;
	uint64_t m_size_bytes;
	uint64_t m_num_entries;

	TwoBitArray(const uint64_t size) {
		m_num_entries = size;
		m_size_bytes = (size + 3) >> 2;
		m_data = new uint8_t[m_size_bytes]();	
	}

	~TwoBitArray() {
		delete[] m_data;
	}

	void resize(const uint64_t size) {
		delete[] m_data;
		m_num_entries = size;
		m_size_bytes = (size + 3) >> 2;
		m_data = new uint8_t[m_size_bytes]();	
	}

	uint64_t size() const {
		return m_num_entries;
	}

	uint8_t get(const uint64_t index) const {
		const uint64_t offset = index >> 2;
		const uint64_t remainder = (index & 0b11) << 1;
		return (m_data[offset] >> remainder) & 0b11;
	}

	void set(const uint64_t index, uint8_t val) {
		const uint64_t offset = index >> 2;
		const uint64_t remainder = (index & 0b11) << 1;
		uint8_t existing = m_data[offset];
		existing &= ~(0b11 << remainder);
		existing |= (val & 0b11) << remainder;
		m_data[offset] = existing;
	}

	uint8_t operator[](const uint64_t index) const {
		return get(index);
	}

	void fill(const uint8_t val) {
		const uint8_t masked_val = val & 0b11;

		const uint8_t packed_value = (
			masked_val
			| (masked_val << 2)
			| (masked_val << 4)
			| (masked_val << 6)
		);

		std::fill(m_data, m_data + m_size_bytes, packed_value);
	}
};


// Thank you to DeepSeek-V4.1-Flash-1017 for assembling 
// VCG_MAPPING which I then spot verified.
// graph conventions defined at:
// https://github.com/seung-lab/connected-components-3d/blob/3.2.0/cc3d_graphs.hpp#L73-L92
static constexpr uint8_t VCG_MAPPING[27] = {
	// z = -1
	25, // idx  0: (-1,-1,-1)  -x-y-z   -> bit 26
	17, // idx  1: ( 0,-1,-1)  -y-z     -> bit 18
	24, // idx  2: (+1,-1,-1)  x-y-z    -> bit 25
	15, // idx  3: (-1, 0,-1)  -x-z     -> bit 16
	 5, // idx  4: ( 0, 0,-1)  -z       -> bit 6
	14, // idx  5: (+1, 0,-1)  x-z      -> bit 15
	23, // idx  6: (-1,+1,-1)  -x+y-z   -> bit 24
	16, // idx  7: ( 0,+1,-1)  y-z      -> bit 17
	22, // idx  8: (+1,+1,-1)  +x+y-z   -> bit 23

	// z = 0
	 9, // idx  9: (-1,-1, 0)  -x-y     -> bit 10
	 3, // idx 10: ( 0,-1, 0)  -y       -> bit 4
	 8, // idx 11: (+1,-1, 0)  x-y      -> bit 9
	 1, // idx 12: (-1, 0, 0)  -x       -> bit 2
	26, // idx 13: ( 0, 0, 0)  origin   -> sentinel / unused
	 0, // idx 14: (+1, 0, 0)  +x       -> bit 1
	 7, // idx 15: (-1,+1, 0)  -xy      -> bit 8
	 2, // idx 16: ( 0,+1, 0)  +y       -> bit 3
	 6, // idx 17: (+1,+1, 0)  xy       -> bit 7

	// z = +1
	21, // idx 18: (-1,-1,+1)  -x-y+z   -> bit 22
	13, // idx 19: ( 0,-1,+1)  -yz      -> bit 14
	20, // idx 20: (+1,-1,+1)  +x-y+z   -> bit 21
	11, // idx 21: (-1, 0,+1)  -xz      -> bit 12
	 4, // idx 22: ( 0, 0,+1)  +z       -> bit 5
	10, // idx 23: (+1, 0,+1)  xz       -> bit 11
	19, // idx 24: (-1,+1,+1)  -x+y+z   -> bit 20
	12, // idx 25: ( 0,+1,+1)  yz       -> bit 13
	18  // idx 26: (+1,+1,+1)  xyz      -> bit 19
};

// x,y,z are -1,0,+1
template <std::signed_integral INTEGER>
constexpr uint32_t vcg_mask(const INTEGER x, const INTEGER y, const INTEGER z) {
	return 1u << VCG_MAPPING[(x+1) + 3 * (y+1) + 9 * (z+1)];
}

};

#endif
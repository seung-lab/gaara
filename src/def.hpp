#ifndef __GAARA_DEFINITIONS_HPP__
#define __GAARA_DEFINITIONS_HPP__

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

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
	const uint64_t k_max_size = (1 << 26) >> 3;

	OneBitArray(const uint64_t size) {
		m_num_entries = size;
		m_size = (size + 7) >> 3;
		m_data = new uint8_t[m_size]();
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
	}

	~OneBitArray() {
		delete[] m_data;
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

static OneBitArray simple_lut("lookup_tables/tables/simple.bin");
static OneBitArray isthmus_lut("lookup_tables/tables/isthmus.bin");

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


};

#endif
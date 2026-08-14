Generating Data Files
=====================

The isthmus thinning algorithm by Pal&aacute;gyi avoids repeated computation of connected
components over a 3x3x3 mask by precomputing all possible 2<sup>26</sup>combinations into lookup tables for both the properties "simple" and "isthmus".

On its own, 2^26 is 67 MB, but since each combination is a binary yes/no oracle, each can
be bit packed into about 8 MB.

Here is how to build and use the resulting data files `simple.bin` and `isthmus.bin`.

## Generation

```zsh
cd lookup_tables
make
```

If needed, adjust the C++ compiler in the Makefile. This will run and generate the files. Ordinarily, this will only have to be done once ever unless there's a bug or you change the definition of "simple" based on new insights.

## How to Use

Each file is a bit packed yes (1)/no (0) answer to the question: is this configuration simple or complex?

Given a 3x3x3 mask represented as an array in xyz Fortran order, you can compute i as the bitfield corresponding to whether each voxel is foreground or background, with the 14th to 25th bits inclusive shifted right one unit (as the central pixel is not considered).

This is pseudocode, actually iterating over your real image may be more complex.

```cpp
let img be your binary image.

is_simple_data = read_file("simple.bin");
is_isthmus_data = read_file("isthmus.bin");

unsigned int i = 0;
int ct = 0;
for (int z = 0; z < 3; z++) {
	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++, ct++) {
			if (ct == 13) {
				continue;
			}
			else if (ct < 13) {
				i |= ((img[x + 3 * y + 9 * z] > 0) << ct);
			}
			else {
				i |= ((img[x + 3 * y + 9 * z] > 0) << (ct-1));
			}
		}
	}
}

// manually unpacking the bit packed image
bool is_simple = (is_simple_data[i] >> (i & 0b111)) & 0b1;
bool is_isthmus = (is_isthmus_data[i] >> (i & 0b111)) & 0b1;

printf("simple: %d isthmus: %d\n", is_simple, is_isthmus);
```






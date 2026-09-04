# gaara
Skeleton thinning algorithm for large images.

```python
import gaara
import numpy as np
import fastmorph

arr = np.load(...) # some 3d array in fortran order

# While holes are fine, cavities are non-sensical for
# a topological thinning algorithm. You'll get a result
# containing hulls that can't be thinned, and the 
# performance will suffer. So make sure you have no
# cavities! Your dentist will thank you.
# fastmorph.fill_holes_v2 is efficient on multiple labels.
filled, holes = fastmorph.fill_holes_v2(arr)

# binary images are more efficient to process
# less memory and faster.

# in_place: modify the input array in_place (saves memory)
# binary_image: consider the input image a binary image (foreground/bg only)
#   regardless of values. 0 is background.
# preserve_endpoints: mark endpoints 3x3x3 stencils containing exactly 2 foreground voxels
#   for preservation. By default, the palagyi algorithm is more aggressive.
# radius: at the cost of a full distance transform, also record vertex radii

skeletons, thin_arr = gaara.skeletonize(
    filled,
    binary_image=True,
    in_place=False,
    preserve_endpoints=False,
    radius=True, 
    threads=2, # Threading only works for binary images right now
)

# You can specify points of interest ("anchors") to be preserved in voxel space
skeletons, thin_arr = gaara.skeletonize(filled, anchors=[(10,10,10)])

# Perform the thinning action on the image without extracting skeletons
thin_arr = gaara.thin(filled, binary_image=False, in_place=False, preserve_endpoints=False)

# For images larger than RAM
# requires the "big" extra install dependency
import crackle
compressed_array = crackle.aload("image.ckl")
result_array = gaara.thin_crackle(
    compressed_array,
    binary_image=True,
    preserve_endpoints=False,
    memory=int(20e9), # Set a RAM amount you are comfortable with
    threads = 2, # Go faster, but uses more RAM and so smaller chunks sizes
    padding = 1, # Perform more thinning iterations per a chunk
    verbose = 0, # Levels: 1 (iteration info), 2 (+chunk info), 3 (+timings)
    fill_holes = True, # by default fill holes on each chunk
)

skeletons = gaara.extract_skeletons_crackle(result_array)
```

Gaara is a skeletion generation via voxel thinning algorithm based on Pal&aacute;gyi's 2014 paper [1] and inspired by Matejek et al.'s work on [synapseaware](https://github.com/Rhoana/synapseaware/). [2] Palágyi and Németh's work on endpoint preservation is highly similar to this algorithm when `preserve_endpoints` is enabled (for some reason, there is no isthmus table in their 2019 version). [4]

While efficient and fine for tracing neurites that have a natural tree structure, a shortcoming of [Kimimaro](https://github.com/seung-lab/kimimaro) is its inability to generate topologically correct skeletons [3] which have applications to e.g. glia and blood vessels.

Gaara attempts to be an efficient voxel thinning algorithm implementation that can handle very large images on a single machine by making use of crackle compression dynamically.

## Installation

The PyPI page is located [here](https://pypi.org/project/gaara/).

```bash
pip install "gaara[big]"
```

The optional "big" dependency will enable support for CrackleArray skeletonization, which supports volumes larger than RAM.

If you are installing from source, ensure the lookup tables have been generated (they should be stored in version control). You can regenerate them using `make` in the lookup_tables directory. It takes about a minute to compute them.

Note that the Python version uses static linking using generated `.hpp` files that is triggered by the C++ macro `GAARA_STATICALLY_LINK_LUTS`. This inflates the size of the binary by ~16MB.

## References

1. K. Palágyi et al., “A Sequential 3D Thinning Algorithm and Its Medical Applications,” in Information Processing in Medical Imaging, M. F. Insana and R. M. Leahy, Eds., Berlin, Heidelberg: Springer, 2001, pp. 409–415. doi: 10.1007/3-540-45729-1_42.  

2. B. Matejek, D. Wei, X. Wang, J. Zhao, K. Palágyi, and H. Pfister, “Synapse-Aware Skeleton Generation for Neural Circuits,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2019, D. Shen, T. Liu, T. M. Peters, L. H. Staib, C. Essert, S. Zhou, P.-T. Yap, and A. Khan, Eds., Cham: Springer International Publishing, 2019, pp. 227–235. doi: 10.1007/978-3-030-32239-7_26.  

3. T. A. Syed, M. Youssef, A. L. Schober, Y. Kubota, K. K. Murai, and C. K. Salmon, “Beyond Neurons: Computer Vision Methods for Analysis of Morphologically Complex Astrocytes,” Frontiers in Computer Science, vol. 6, Sep. 2024, doi: 10.3389/fcomp.2024.1156204.  

4. K. Palágyi and G. Németh, “Centerline Extraction from 3D Airway Trees Using Anchored Shrinking,” in Advances in Visual Computing, G. Bebis, R. Boyle, B. Parvin, D. Koracin, D. Ushizima, S. Chai, S. Sueda, X. Lin, A. Lu, D. Thalmann, C. Wang, and P. Xu, Eds., Cham: Springer International Publishing, 2019, pp. 419–430. doi: 10.1007/978-3-030-33723-0_34.


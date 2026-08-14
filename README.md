# gaara
Skeleton thinning algorithm for large images.

Gaara is a skeletion generation via voxel thinning algorithm based on Pal&aacute;gyi's 2014 paper [1] and inspired by Matejek et al.'s work on [synapseaware](https://github.com/Rhoana/synapseaware/). [2]

While efficient and fine for tracing neurites that have a natural tree structure, a shortcoming of [Kimimaro](https://github.com/seung-lab/kimimaro) is its inability to generate topologically correct skeletons [3] which have applications to e.g. glia and blood vessels.

Gaara attempts to be an efficient voxel thinning algorithm implementation that can handle very large images on a single machine by making use of crackle compression dynamically.

## References

1. K. Palágyi et al., “A Sequential 3D Thinning Algorithm and Its Medical Applications,” in Information Processing in Medical Imaging, M. F. Insana and R. M. Leahy, Eds., Berlin, Heidelberg: Springer, 2001, pp. 409–415. doi: 10.1007/3-540-45729-1_42.  

2. B. Matejek, D. Wei, X. Wang, J. Zhao, K. Palágyi, and H. Pfister, “Synapse-Aware Skeleton Generation for Neural Circuits,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2019, D. Shen, T. Liu, T. M. Peters, L. H. Staib, C. Essert, S. Zhou, P.-T. Yap, and A. Khan, Eds., Cham: Springer International Publishing, 2019, pp. 227–235. doi: 10.1007/978-3-030-32239-7_26.  

3. T. A. Syed, M. Youssef, A. L. Schober, Y. Kubota, K. K. Murai, and C. K. Salmon, “Beyond Neurons: Computer Vision Methods for Analysis of Morphologically Complex Astrocytes,” Frontiers in Computer Science, vol. 6, Sep. 2024, doi: 10.3389/fcomp.2024.1156204.  

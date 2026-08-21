from typing import Union, Iterable, Any

import numpy as np
import numpy.typing as npt

import osteoid

from . import fastgaara

__all__ = ["thin", "skeletonize"]

def thin(
    labels:npt.NDArray[np.integer],
    binary_image:bool = False,
    in_place:bool = False,
    preserve_endpoints:bool = False,
    max_iterations:int = -1,
) -> npt.NDArray[np.integer]:
    """
    Apply Palagyi's 3D voxel thinning algorithm to `labels`, a binary image.

    in_place: modify the input array in_place (saves memory)
    binary_image: consider the input image a binary image (foreground/bg only)
      regardless of values. 0 is background.
    preserve_endpoints: mark endpoints 3x3x3 stencils containing exactly 2 foreground voxels
      for preservation. By default, the palagyi algorithm is more aggressive. It will mark
      "ismuths" for preservation, which erodes the ends a little bit.

    Reference:

    K. Palágyi, "A Sequential 3D Curve-Thinning Algorithm Based on Isthmuses,"
    in Advances in Visual Computing, vol. 8888,
    G. Bebis, R. Boyle, B. Parvin, D. Koracin, R. McMahan, J. Jerald, 
    H. Zhang, S. M. Drucker, C. Kambhamettu, M. El Choubassi, Z. Deng, 
    and M. Carlson, Eds., 
    Cham: Springer International Publishing, 2014, pp. 406–415.
    doi: 10.1007/978-3-319-14364-4_39.
    """
    if labels.ndim != 3:
        raise ValueError(f"This function only supports 3D images. Got: {labels.shape}")

    if labels.size <= 1:
        return labels

    if in_place and not labels.flags.writeable:
        raise ValueError("Cannot perform an in-place operation on non-writeable data.")
    elif in_place and not labels.flags.f_contiguous:
        raise ValueError("Cannot perform an in-place operation on non-Fortran ordered data.")
    elif not in_place:
        labels = np.copy(labels, order="F")

    orig_dtype = labels.dtype
    if labels.dtype == bool:
        binary_image = True
        labels = labels.view(np.uint8)

    if binary_image:
        fastgaara.thin_binary(labels, preserve_endpoints, max_iterations)
    else:
        fastgaara.thin_multilabel(labels, preserve_endpoints, max_iterations)

    return labels

def skeletonize(
    labels:npt.NDArray[np.integer],
    binary_image:bool = False,
    in_place:bool = False,
    preserve_endpoints:bool = False,
) -> tuple[osteoid.Skeleton|dict[int,osteoid.Skeleton], npt.NDArray[np.integer]]:
    """
    Apply Palagyi's 3D voxel thinning algorithm to `labels`, a binary image
    and return skeletons.

    in_place: modify the input array in_place (saves memory)
    binary_image: consider the input image a binary image (foreground/bg only)
      regardless of values. 0 is background.
    preserve_endpoints: mark endpoints 3x3x3 stencils containing exactly 2 foreground voxels
      for preservation. By default, the palagyi algorithm is more aggressive. It will mark
      "ismuths" for preservation, which erodes the ends a little bit.

    Reference:

    K. Palágyi, "A Sequential 3D Curve-Thinning Algorithm Based on Isthmuses,"
    in Advances in Visual Computing, vol. 8888,
    G. Bebis, R. Boyle, B. Parvin, D. Koracin, R. McMahan, J. Jerald, 
    H. Zhang, S. M. Drucker, C. Kambhamettu, M. El Choubassi, Z. Deng, 
    and M. Carlson, Eds., 
    Cham: Springer International Publishing, 2014, pp. 406–415.
    doi: 10.1007/978-3-319-14364-4_39.
    """
    if labels.ndim != 3:
        raise ValueError(f"This function only supports 3D images. Got: {labels.shape}")

    if labels.size <= 1:
        return labels

    if in_place and not labels.flags.writeable:
        raise ValueError("Cannot perform an in-place operation on non-writeable data.")
    elif in_place and not labels.flags.f_contiguous:
        raise ValueError("Cannot perform an in-place operation on non-Fortran ordered data.")
    elif not in_place:
        labels = np.copy(labels, order="F")

    orig_dtype = labels.dtype
    if labels.dtype == bool:
        binary_image = True
        labels = labels.view(np.uint8)

    if binary_image:
        (vertices, edges) = fastgaara.skeletonize_binary(labels, preserve_endpoints)
        skeletons = osteoid.Skeleton(vertices, edges)
    else:
        skeletons = fastgaara.skeletonize_multilabel(labels, preserve_endpoints)
        skeletons = { 
            segid: osteoid.Skeleton(vertices, edges) 
            for segid, (vertices, edges) in skeletons.items()
        }

    return (skeletons, labels)
    
def thin_crackle(
    labels:Union["CrackleArray", bytes],
    binary_image:bool = False,
    in_place:bool = False,
    preserve_endpoints:bool = False,
    memory:int = int(8e9),
    threads:int = -1,
) -> "CrackleArray":
    """
    Apply Palagyi's 3D voxel thinning algorithm to a CrackleArray.

    See https://github.com/seung-lab/crackle

    in_place: modify the input array in_place (saves memory)
    binary_image: consider the input image a binary image (foreground/bg only)
      regardless of values. 0 is background.
    preserve_endpoints: mark endpoints 3x3x3 stencils containing exactly 2 foreground voxels
      for preservation. By default, the palagyi algorithm is more aggressive. It will mark
      "ismuths" for preservation, which erodes the ends a little bit.

    Reference:

    K. Palágyi, "A Sequential 3D Curve-Thinning Algorithm Based on Isthmuses,"
    in Advances in Visual Computing, vol. 8888,
    G. Bebis, R. Boyle, B. Parvin, D. Koracin, R. McMahan, J. Jerald, 
    H. Zhang, S. M. Drucker, C. Kambhamettu, M. El Choubassi, Z. Deng, 
    and M. Carlson, Eds., 
    Cham: Springer International Publishing, 2014, pp. 406–415.
    doi: 10.1007/978-3-319-14364-4_39.
    """
    import crackle
    from crackle import CrackleArray

    if isinstance(labels, bytes):
        labels = CrackleArray(labels)
    elif not isinstance(labels, CrackleArray):
        raise ValueError("This function only accepts type CrackleArrays.")

    if labels.ndim != 3:
        raise ValueError(f"This function only supports 3D images. Got: {labels.shape}")

    if labels.size <= 1:
        return labels

    header = labels.header()

    if (header.data_width * header.voxels() < memory):
        labels = thin(
            labels.numpy(),
            binary_image=binary_image,
            preserve_endpoints=preserve_endpoints,
            in_place=True,
        )
        return crackle.compressa(labels, parallel=threads)

    slice_memory = header.data_width * header.sx * header.sy 
    cz = (memory // slice_memory) - 1
    cz = max(cz, 4)
    
    if (cz * slice_memory > memory):
        raise MemoryError(
            f"Not enough memory to process this volume, try increasing the limit. "
            f"Limit: {memory} bytes. "
            f"Per slice memory usage: {slice_memory} bytes."
        )

    num_chunks = int(np.ceil(header.sz / cz))
    num_chunks = max(num_chunks, 1)

    compressed_chunks = [ None ] * num_chunks
    num_deleted_points = np.zeros([num_chunks], dtype=np.uint64)

    iterated_labels = labels.asfortranarray()
    iterated_labels.parallel = threads
    
    while True:
        for i in range(num_chunks):
            z = i * cz

            chunk_start = std::max(0, z-1)
            chunk_end = z + cz + 1
            chunk_end = min(sz, chunk_end)

            arr = iterated_labels[:,:,chunk_start:chunk_end]
            num_deleted_points[i] = thin(
                arr, 
                binary_image=binary_image,
                preserve_endpoints=preserve_endpoints,
                max_iterations=1,
            )
            
            if chunk_start != 0:
                arr = arr[:,:,1:]
            if chunk_end != header.sz:
                arr = arr[:,:,:-1]

            compressed_chunks[i] = crackle.compressa(arr, parallel=threads)

        iterated_labels = crackle.zstack(compressed_chunks)
        compressed_chunks = [ None ] * num_chunks

        if np.sum(num_deleted_points) == 0:
            break

    return iterated_labels



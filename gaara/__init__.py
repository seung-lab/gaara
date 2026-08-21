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
    return_num_deleted_points:bool = False,
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
        num_deleted_points = fastgaara.thin_binary(labels, preserve_endpoints, max_iterations)
    else:
        num_deleted_points = fastgaara.thin_multilabel(labels, preserve_endpoints, max_iterations)

    if return_num_deleted_points:
        return (labels, num_deleted_points)
    else:
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
    threads:int = 0,
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
    import time
    import psutil
    import multiprocessing as mp

    if isinstance(labels, bytes):
        labels = CrackleArray(labels)
    elif not isinstance(labels, CrackleArray):
        raise ValueError("This function only accepts type CrackleArrays.")

    if labels.ndim != 3:
        raise ValueError(f"This function only supports 3D images. Got: {labels.shape}")

    if labels.size <= 1:
        return labels

    if threads == 0:
        threads = mp.cpu_count()

    header = labels.header()
    slice_memory = header.data_width * header.sx * header.sy 
    # Estimate of crackle per-slice decoding memory usage
    decoding_memory = (header.data_width + 4) * slice_memory * threads # ccl + slice 

    estimated_memory_requirement = header.data_width * header.voxels() + decoding_memory
    estimated_memory_requirement += len(labels) * 2

    system_memory = psutil.virtual_memory().available

    if estimated_memory_requirement < memory and estimated_memory_requirement < system_memory:
        print("lall: ", header.data_width * header.voxels())
        labels = thin(
            labels.numpy(),
            binary_image=binary_image,
            preserve_endpoints=preserve_endpoints,
            in_place=True,
        )
        return crackle.compressa(labels, parallel=threads)

    cz = (memory // decoding_memory) - 2
    cz = max(cz, 4)

    estimated_memory_requirement = (cz+2) * slice_memory + decoding_memory
    estimated_memory_requirement += len(labels) * 3

    if estimated_memory_requirement > min(memory, system_memory):
        raise MemoryError(
            f"Not enough memory to process this volume, try increasing the limit or reducing the number of threads.\n"
            f"User Limit: {memory} bytes\n"
            f"Available: {system_memory} bytes\n"
            f"Estimated memory requirement: {estimated_memory_requirement / 1e9:.1f} GB."
        )
    
    compressed_chunks = []
    num_deleted_points = []

    iterated_labels = labels.asfortranarray()
    iterated_labels.parallel = threads
    
    print(cz, slice_memory, memory)

    num_iters = 0

    while True:
        i = 0
        while True:
            z = (i * cz) - 1
            chunk_start = max(0, z-1)
            chunk_end = z + cz + 1
            chunk_end = min(header.sz, chunk_end)

            print("z=",z, chunk_start, chunk_end)

            s = time.perf_counter()
            arr = iterated_labels[:,:,chunk_start:chunk_end]
            print(f"decompress {time.perf_counter() - s:.3f}s")

            s = time.perf_counter()
            arr, N = thin(
                arr, 
                binary_image=binary_image,
                preserve_endpoints=preserve_endpoints,
                in_place=True,
                max_iterations=1,
                return_num_deleted_points=True,
            )
            print(f"thinning {time.perf_counter() - s:.3f}s")
            num_deleted_points.append(N)
            
            if chunk_start != 0:
                arr = arr[:,:,1:]
            if chunk_end != header.sz:
                arr = arr[:,:,:-1]

            s = time.perf_counter()
            compressed_chunks.append(
                crackle.compressa(arr, parallel=threads)
            )
            print(f"compress {time.perf_counter() - s:.3f}s")
            del arr

            i += 1

            if chunk_end >= header.sz:
                break

        iterated_labels = CrackleArray(crackle.zstack(compressed_chunks))
        iterated_labels.parallel = threads
        
        points_deleted_this_round = np.sum(num_deleted_points)
        num_iters += 1
        
        compressed_chunks = []
        num_deleted_points = []

        print(f"deleted={points_deleted_this_round}, num_iters={num_iters}")

        if points_deleted_this_round == 0:
            break

    return iterated_labels



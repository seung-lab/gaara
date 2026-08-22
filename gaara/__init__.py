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
    preserve_endpoints:bool = False,
    memory:int = -1,
    threads:int = 0,
    padding:int = 1,
    verbose:int = 0,
    fill_holes:bool = True,
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
    import multiprocessing as mp
    import time

    import crackle
    from crackle import CrackleArray
    import fill_voids
    import psutil

    assert padding > 0 and int(padding) == padding
    assert threads >= 0

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

    labels = labels.asfortranarray()
    header = labels.header()

    slice_memory = header.data_width * header.sx * header.sy 
    # Estimate of crackle per-slice decoding memory usage
    decoding_memory = (header.data_width + 4) * slice_memory * threads # ccl + slice 

    estimated_memory_requirement = header.data_width * header.voxels()
    estimated_memory_requirement *= 1.2 # for algorithm interal data structures
    estimated_memory_requirement += decoding_memory
    estimated_memory_requirement += len(labels) * 3
    estimated_memory_requirement = int(estimated_memory_requirement)

    system_memory = psutil.virtual_memory().available
    if memory < 0:
        memory = system_memory

    if estimated_memory_requirement < memory and estimated_memory_requirement < system_memory:
        if verbose:
            print("Processing all at once.")
            print(f"Estimated Memory Requirement: {estimated_memory_requirement} bytes")
        labels = thin(
            labels.numpy(),
            binary_image=binary_image,
            preserve_endpoints=preserve_endpoints,
            in_place=True,
        )
        return crackle.compressa(labels, parallel=threads)

    estimated_memory_requirement = decoding_memory + len(labels) * 3

    internal_factor = 1.5 # for thinning data structures

    cz = ((memory - estimated_memory_requirement) / internal_factor / slice_memory) - 2 * padding
    cz = max(cz, 1)
    cz = min(cz, header.sz)
    cz = int(cz)

    estimated_memory_requirement += internal_factor * min((cz + 2 * padding), header.sz) * slice_memory
    estimated_memory_requirement = int(estimated_memory_requirement)

    if estimated_memory_requirement > min(memory, system_memory):
        raise MemoryError(
            f"Not enough memory to process this volume, try increasing the limit, reducing the number of threads, or reducing padding.\n"
            f"User Limit: {memory} bytes\n"
            f"Available: {system_memory} bytes\n"
            f"Estimated memory requirement: {estimated_memory_requirement / 1e9:.1f} GB."
        )
    
    compressed_chunks = []
    num_deleted_points = []

    iterated_labels = labels
    iterated_labels.parallel = threads
    
    if verbose:
        print(f"chunk size + padding: {cz + 2 * padding}")
        print(f"Estimated Memory Requirement: {estimated_memory_requirement} bytes")

    num_iters = 0

    while True:
        if verbose:
            print(f"Iteration {num_iters}")

        i = 0
        while True:
            z = (i * cz)
            if z == 0:
                chunk_start = 0
            else:
                chunk_start = z - padding

            chunk_end = z + cz + padding
            chunk_end = min(header.sz, chunk_end)

            if verbose > 1:
                print(f"z = {chunk_start}:{chunk_end}")

            s = time.perf_counter()
            arr = iterated_labels[:,:,chunk_start:chunk_end]
            e = time.perf_counter()

            if verbose > 2:
                print(f"decompress: {e - s:.2f}s")

            if fill_holes:
                s = time.perf_counter()
                arr = fill_voids.fill(arr, in_place=True)
                e = time.perf_counter()

                if verbose > 2:
                    print(f"fill_voids: {e - s:.2f}s")

            s = time.perf_counter()
            arr, N = thin(
                arr, 
                binary_image=binary_image,
                preserve_endpoints=preserve_endpoints,
                in_place=True,
                max_iterations=padding,
                return_num_deleted_points=True,
            )
            e = time.perf_counter()
            if verbose > 2:
                print(f"thinning: {e - s:.2f}s")

            num_deleted_points.append(N)
            
            if chunk_start != 0:
                arr = arr[:,:,padding:]
            if chunk_end != header.sz:
                arr = arr[:,:,:-padding]

            s = time.perf_counter()
            compressed_chunks.append(
                crackle.compressa(arr, parallel=threads)
            )
            e = time.perf_counter()
            if verbose > 2:
                print(f"compress: {e - s:.2f}s")
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

        if verbose:
            print(f"points deleted: {points_deleted_this_round}")

        if points_deleted_this_round == 0:
            break

    return iterated_labels


def extract_skeletons_crackle(
    labels:Union["CrackleArray", bytes],
    binary_image:bool = False,
    preserve_endpoints:bool = False,
    memory:int = -1,
    threads:int = 0,
    padding:int = 1,
    verbose:int = 0,
) -> dict[int, osteoid.Skeleton]:
    import crackle
    from crackle import CrackleArray
    import time
    import psutil
    import multiprocessing as mp

    assert threads >= 0

    if isinstance(labels, bytes):
        labels = CrackleArray(labels)
    elif not isinstance(labels, CrackleArray):
        raise ValueError("This function only accepts type CrackleArrays.")

    if labels.ndim != 3:
        raise ValueError(f"This function only supports 3D images. Got: {labels.shape}")

    if threads == 0:
        threads = mp.cpu_count()

    labels = labels.asfortranarray()

    slice_memory = header.data_width * header.sx * header.sy 
    # Estimate of crackle per-slice decoding memory usage
    decoding_memory = (header.data_width + 4) * slice_memory * threads # ccl + slice 

    system_memory = psutil.virtual_memory().available
    if memory < 0:
        memory = system_memory

    internal_factor = 1.1 # for extraction data structures

    estimated_memory_requirement = header.data_width * header.voxels()
    estimated_memory_requirement *= internal_factor # for algorithm interal data structures
    estimated_memory_requirement += decoding_memory
    estimated_memory_requirement += len(labels) + len(thinned_labels)
    estimated_memory_requirement = int(estimated_memory_requirement)

    if estimated_memory_requirement < memory and estimated_memory_requirement < system_memory:
        if verbose:
            print("Processing all at once.")
            print(f"Estimated Memory Requirement: {estimated_memory_requirement} bytes")
        
        return fastgaara.extract_skeletons(labels.numpy())

    estimated_memory_requirement = decoding_memory + len(labels) + len(thinned_labels)

    cz = ((memory - estimated_memory_requirement) / internal_factor / slice_memory)
    cz = max(cz, 1)
    cz = min(cz, header.sz)
    cz = int(cz)

    estimated_memory_requirement += internal_factor * min(cz, header.sz) * slice_memory
    estimated_memory_requirement = int(estimated_memory_requirement)

    if estimated_memory_requirement > min(memory, system_memory):
        raise MemoryError(
            f"Not enough memory to process this volume, try increasing the limit, reducing the number of threads, or reducing padding.\n"
            f"User Limit: {memory} bytes\n"
            f"Available: {system_memory} bytes\n"
            f"Estimated memory requirement: {estimated_memory_requirement / 1e9:.1f} GB."
        )

    skeletons = {}

    num_chunks = int(np.ceil(header.sz / cz))
    for i in range(num_chunks):
        z_start = i * cz
        z_end = min((i+1) * cz, header.sz)
        arr = thinned_labels[:,:, z_start:z_end ]
        arr_skeletons = fastgaara.extract_skeletons(arr)
        del arr

        for segid in arr_skeletons.keys():
            if segid not in skeletons:
                skeletons[segid] = arr_skeletons[segid]
                continue

            skeletons[segid] = osteoid.Skeleton.simple_merge([ 
                skeletons[segid], arr_skeletons[segid] 
            ])

    for segid in skeletons:
        skel = skeletons[segid]
        skeletons[segid] = skel.consolidate()

    return skeletons



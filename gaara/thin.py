from typing import Union, Iterable, Any, Optional
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

import crackle
from crackle import CrackleArray
import edt
import osteoid
from osteoid import Skeleton

from . import fastgaara
from . import analyze

__all__ = [
    "thin",
    "skeletonize",
    "extract_skeletons",
    "thin_crackle",
    "extract_skeletons_crackle",
    "skeletonize_crackle",
]

def extract_skeletons(
    img:np.ndarray,
    voxel_graph:Optional[npt.NDArray[np.uint32]] = None
) -> dict[int, Skeleton]:
    """Extract all skeletons from the thinned input array."""
    skeletons = fastgaara.extract_skeletons(img, None)
    return { 
        segid: Skeleton(vertices, edges, segid=segid, default_attributes=False)
        for segid, (vertices, edges) in skeletons.items()
    }

def thin(
    labels:npt.NDArray[np.integer],
    binary_image:bool = False,
    in_place:bool = False,
    preserve_endpoints:bool = False,
    anchors:npt.NDArray[np.uint16] = np.empty((0, 3), dtype=np.uint16),
    max_iterations:int = -1,
    return_num_deleted_points:bool = False,
    threads:int = 1,
    revisit:bool = False,
    final_mask:bool = True,
    voxel_graph:Optional[npt.NDArray[np.uint32]] = None,
) -> npt.NDArray[np.integer]:
    """
    Apply Palagyi's 3D voxel thinning algorithm to `labels`, a binary image.

    in_place: modify the input array in_place (saves memory)
    binary_image: consider the input image a binary image (foreground/bg only)
      regardless of values. 0 is background.
    preserve_endpoints: mark endpoints 3x3x3 stencils containing exactly 2 foreground voxels
      for preservation. By default, the palagyi algorithm is more aggressive. It will mark
      "ismuths" for preservation, which erodes the ends a little bit.
    max_iterations: restrict the algorithm to this many iterations. 
        This is useful for debugging or designing chunked versions of the algorithm.
    anchors: a list of (x,y,z) tuples that should be preserved.
    threads: number of threads to use (binary images only)
    return_num_deleted_points: if true, return a tuple (array, int) where
        the second element is the number of voxels that were deleted.
    voxel_graph: 26-connected voxel connectivity graph based on 
        cc3d's format. This is useful for distangling autapses.

    (internal options)
    The following two options are for the chunked version of the algorithm
    and are likely not useful to end users.

    revisit: skip initial masking and calculate border points based by examining
        the pre-exisitng voxel coloration.
    final_mask: do not erase the voxel states after processing

    References:

    K. Palágyi, "A Sequential 3D Curve-Thinning Algorithm Based on Isthmuses,"
    in Advances in Visual Computing, vol. 8888,
    G. Bebis, R. Boyle, B. Parvin, D. Koracin, R. McMahan, J. Jerald, 
    H. Zhang, S. M. Drucker, C. Kambhamettu, M. El Choubassi, Z. Deng, 
    and M. Carlson, Eds., 
    Cham: Springer International Publishing, 2014, pp. 406–415.
    doi: 10.1007/978-3-319-14364-4_39.

    K. Palágyi and G. Németh, “Centerline Extraction from 3D Airway Trees Using Anchored Shrinking,”
    in Advances in Visual Computing, G. Bebis, R. Boyle, B. Parvin, D. Koracin, D. Ushizima, 
    S. Chai, S. Sueda, X. Lin, A. Lu, D. Thalmann, C. Wang, and P. Xu, Eds., 
    Cham: Springer International Publishing, 2019, pp. 419–430. 
    doi: 10.1007/978-3-030-33723-0_34.
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

    anchors = np.asarray(anchors).astype(np.uint16, copy=False)
    if anchors is not None and (anchors.ndim != 2 or anchors.shape[1] != 3):
        raise ValueError(f"Preserve must be an Nx3 array of voxel coordinates.")

    if binary_image:
        num_deleted_points = fastgaara.thin_binary(
            labels, 
            preserve_endpoints,
            anchors,
            max_iterations,
            threads,
            revisit,
            final_mask,
            voxel_graph
        )
    elif voxel_graph is not None:
        raise NotImplementedError("voxel_graph is not yet supported for multilabel images.")
    else:
        num_deleted_points = fastgaara.thin_multilabel(
            labels,
            preserve_endpoints,
            anchors,
            max_iterations
        )

    if return_num_deleted_points:
        return (labels, num_deleted_points)
    else:
        return labels

def skeletonize(
    labels:npt.NDArray[np.integer],
    binary_image:bool = False,
    in_place:bool = False,
    preserve_endpoints:bool = False,
    anchors:npt.NDArray[np.uint16] = np.empty((0, 3), dtype=np.uint16),
    radius:bool = True,
    cross_section:int = 6,
    anisotropy:tuple[float,float,float] = (1.0, 1.0, 1.0),
    threads:int = 1,
    voxel_graph:Optional[npt.NDArray[np.uint32]] = None,
) -> tuple[Skeleton|dict[int,Skeleton], npt.NDArray[np.integer]]:
    """
    Apply Palagyi's 3D voxel thinning algorithm to `labels`, a binary image
    and return skeletons.

    binary_image: consider the input image a binary image (foreground/bg only)
      regardless of values. 0 is background.
    in_place: modify the input array in_place (saves memory)
    preserve_endpoints: mark endpoints 3x3x3 stencils containing exactly 2 foreground voxels
      for preservation. By default, the palagyi algorithm is more aggressive. It will mark
      "ismuths" for preservation, which erodes the ends a little bit.
    anchors: an Nx3 array of integer coordinates that tell the thinning algorithm
        to preserve the indicated voxels.
    radius: compute the euclidean (L2) distance transfrom and assign the values
        to the "radius" parameter of the skeleton
    cross_section: if >0, adds cross section analysis.
        The integer is the number of points to perform average smoothing
        over.
    threads: number of threads to use, 0 means all cores
    voxel_graph: 26-connected voxel connectivity graph based on 
        cc3d's format. This is useful for distangling autapses.
        This is only supported for binary images currently.

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

    anchors = np.asarray(anchors).astype(np.uint16, copy=False)
    if anchors is not None and (anchors.ndim != 2 or anchors.shape[1] != 3):
        raise ValueError(f"Preserve must be an Nx3 array of voxel coordinates.")

    if radius or cross_section > 0:
        original_labels = crackle.compressa(labels, parallel=threads)

    orig_dtype = labels.dtype
    if labels.dtype == bool:
        binary_image = True
        labels = labels.view(np.uint8)

    if binary_image:
        (vertices, edges) = fastgaara.skeletonize_binary(
            labels, preserve_endpoints, anchors, threads, voxel_graph,
        )
        skeletons = { 1: Skeleton(vertices, edges, default_attributes=False, segid=1) }
    elif voxel_graph is not None:
        raise NotImplementedError("voxel_graph is not yet implemented for multilabel images.")
    else:
        skeletons = fastgaara.skeletonize_multilabel(
            labels, preserve_endpoints, anchors
        )
        skeletons = { 
            segid: Skeleton(vertices, edges, default_attributes=False, segid=segid) 
            for segid, (vertices, edges) in skeletons.items()
        }

    # save some space before computing the edt which requires
    # a copy + a distance transform
    if radius or cross_section > 0:
        labels = crackle.compressa(labels, parallel=threads)

    if radius:
        dt = edt.edt(
            original_labels.numpy(),
            anisotropy=anisotropy,
            parallel=threads,
            voxel_graph=voxel_graph,
        )

        for skel in skeletons.values():
            verts = skel.vertices.astype(np.uint16)
            if len(verts) == 0:
                continue

            vradii = dt[verts[:,0], verts[:,1], verts[:, 2]]
            p = skel.add_vertex_attribute('radius', vradii)

        dt = None

    if cross_section > 0:
        skeletons = analyze.cross_sectional_area(
            original_labels.numpy(),
            skeletons,
            anisotropy=anisotropy,
            smoothing_window=cross_section,
        )

    if anisotropy != (1.0, 1.0, 1.0):
        for skel in skeletons.values():
            skel.vertices *= np.array(anisotropy, dtype=np.float32)

    if radius or cross_section > 0:
        labels = labels.numpy()

    if binary_image:
        return (skeletons[1], labels)
    else:
        return (skeletons, labels)

def thin_crackle(
    labels:Union["CrackleArray", bytes],
    binary_image:bool = True,
    preserve_endpoints:bool = False,
    memory:int = -1,
    threads:int = 0,
    padding:int = 6,
    verbose:int = 0,
    checkpoints:int = 0,
    require_chunking:bool = False,
    voxel_graph_fn:Optional[Callable[[int,int], npt.NDArray[np.uint32]]] = None,
) -> CrackleArray:
    """
    Apply Palagyi's 3D voxel thinning algorithm to a CrackleArray.

    See https://github.com/seung-lab/crackle

    in_place: modify the input array in_place (saves memory)
    binary_image: consider the input image a binary image (foreground/bg only)
      regardless of values. 0 is background.
    preserve_endpoints: mark endpoints 3x3x3 stencils containing exactly 2 foreground voxels
      for preservation. By default, the palagyi algorithm is more aggressive. It will mark
      "ismuths" for preservation, which erodes the ends a little bit.
    
    checkpoints: save "checkpoint.ckl" every xth iteration. 0 means no checkpoints.
    require_chunking: for testing, force using the chunked path even if there is 
        enough memory to do it all in one shot.
    voxel_graph_fn: A function taking z_start and z_end that produces a 26-connected
        uint32 voxel connectivity graph (VCG) in the form given by cc3d. The x and y
        dimensions must match the input labels.

        The VCG is used to disambiguate self-touches (autapses) that would otherwise
        appear to be a single contiguous image. VCG is currently only supported for
        binary images.

        Using a VCG will increase the memory requirement.

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

    import psutil

    assert padding > 0 and int(padding) == padding and padding % 6 == 0
    assert threads >= 0

    if binary_image == False:
        raise ValueError("multilabel chunked thinning is not yet supported.")

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

    sxy = header.sx * header.sy 
    slice_memory = sxy * header.data_width
    # Estimate of crackle per-slice decoding memory usage
    decoding_memory = (header.data_width + 4) * sxy * threads # uint32 ccl + slice 
    estimated_memory_requirement = header.data_width * header.voxels()
    estimated_memory_requirement *= 1.2 # for algorithm interal data structures
    estimated_memory_requirement += decoding_memory
    estimated_memory_requirement += len(labels) * 3
    estimated_memory_requirement = int(estimated_memory_requirement)

    if voxel_graph_fn is not None:
        estimated_memory_requirement += 4 * header.voxels() # uint32 vcg

    system_memory = psutil.virtual_memory().total

    if memory < 0:
        memory = system_memory

    if not require_chunking and estimated_memory_requirement < memory and estimated_memory_requirement < system_memory:
        if verbose:
            print("Processing all at once.")
            print(f"Estimated Memory Requirement: {estimated_memory_requirement} bytes")
        labels = thin(
            labels.numpy(),
            binary_image=binary_image,
            preserve_endpoints=preserve_endpoints,
            in_place=True,
            threads=threads,
        )
        return crackle.compressa(labels, parallel=threads)

    estimated_memory_requirement = decoding_memory + len(labels) * 3

    internal_factor = 1.5 # for thinning data structures

    cz_slice_memory = slice_memory
    if voxel_graph_fn is not None:
        cz_slice_memory += 4 * sxy

    cz = ((memory - estimated_memory_requirement) / internal_factor / cz_slice_memory) - 2 * padding
    cz = max(cz, 1)
    cz = min(cz, header.sz)
    cz = int(cz)

    estimated_memory_requirement += internal_factor * min((cz + 2 * padding), header.sz) * slice_memory

    if voxel_graph_fn is not None:
        estimated_memory_requirement += min((cz + 2 * padding), header.sz) * sxy * 4 # vcg uint32 range

    estimated_memory_requirement = int(estimated_memory_requirement)

    if estimated_memory_requirement > min(memory, system_memory):
        raise MemoryError(
            f"Not enough memory to process this volume, try increasing the limit, reducing the number of threads, or reducing padding.\n"
            f"User Limit: {memory} bytes\n"
            f"Available: {system_memory} bytes\n"
            f"Estimated memory requirement: {estimated_memory_requirement / 1e9:.1f} GB."
        )
    
    num_chunks = max(int(np.ceil(header.sz / cz)), 0)

    compressed_chunks = [ None ] * num_chunks
    num_deleted_points = np.zeros([num_chunks], dtype=np.int64)
    
    needs_work = np.ones([num_chunks], dtype=bool)
    needs_work_next_round = np.ones([num_chunks], dtype=bool)

    iterated_labels = labels
    iterated_labels.parallel = threads
    
    if verbose:
        print(f"chunk size + padding: {cz + 2 * padding}")
        print(f"Estimated Memory Requirement: {estimated_memory_requirement} bytes")

    num_iters = 0
    vcg = None

    while True:
        if verbose:
            print(f"Iteration {num_iters}")

        for i in range(num_chunks):
            if num_iters > 0 and needs_work[i] == False:
                continue

            z = (i * cz)
            chunk_start = max(z - padding, 0)

            chunk_end = z + cz + padding
            chunk_end = min(header.sz, chunk_end)

            keep_start = z - chunk_start
            keep_end = min(z + cz, header.sz) - chunk_start

            if verbose > 1:
                print(f"z = {chunk_start}:{chunk_end}")

            s = time.perf_counter()
            arr = iterated_labels[:,:,chunk_start:chunk_end]
            e = time.perf_counter()

            if verbose > 2:
                print(f"decompress: {e - s:.2f}s")

            if voxel_graph_fn is not None:
                s = time.perf_counter()
                vcg = voxel_graph_fn(chunk_start, chunk_end)                
                e = time.perf_counter()

                if verbose > 2:
                    print(f"vcg decode: {e - s:.2f}s")

            # Deletions in the padding shouldn't count so we need
            # to establish our own count.
            s = time.perf_counter()
            N_valid = np.count_nonzero(arr[:,:,keep_start:keep_end])
            arr, N = thin(
                arr, 
                binary_image=binary_image,
                preserve_endpoints=preserve_endpoints,
                in_place=True,
                max_iterations=padding // 6,
                return_num_deleted_points=True,
                threads=threads,
                revisit=(num_iters > 0),
                final_mask=False,
                voxel_graph=vcg,
            )
            N = N_valid - np.count_nonzero(arr[:,:,keep_start:keep_end])
            e = time.perf_counter()
            if verbose > 2:
                print(f"thinning: {e - s:.2f}s")

            vcg = None

            if N > 0 and i > 0:
                needs_work_next_round[i-1] = True

            num_deleted_points[i] = N
            needs_work_next_round[i] |= (N>0)

            if N > 0 and i < num_chunks - 1:
                needs_work_next_round[i+1] = True
            
            arr = arr[:,:,keep_start:keep_end]
            
            s = time.perf_counter()
            compressed_chunks[i] = crackle.compressa(arr, parallel=threads)
            e = time.perf_counter()
            if verbose > 2:
                print(f"compress: {e - s:.2f}s")
            del arr

        iterated_labels = CrackleArray(crackle.zstack(compressed_chunks))
        iterated_labels.parallel = threads
        
        points_deleted_this_round = np.sum(num_deleted_points)
        num_iters += 1
        
        if checkpoints > 0 and num_iters % checkpoints == 0:
            iterated_labels.save("checkpoint.ckl")
            with open("checkpoints.txt", "at") as f:
                f.write(f"checkpoint {num_iters} ; {time.time():.1f}\n")
            if verbose:
                print(f"saved checkpoint")

        num_deleted_points[:] = 0

        if verbose:
            print(f"points deleted: {points_deleted_this_round}")

        any_work_remaining = np.any(needs_work_next_round)
        needs_work = np.copy(needs_work_next_round)
        needs_work_next_round[:] = False

        if not any_work_remaining:
            break

    # b/c we are skipping final mask in thin, we need to 
    # do the masking manually here with "> 0"
    return iterated_labels > 0


def extract_skeletons_crackle(
    labels:Union["CrackleArray", bytes],
    memory:int = -1,
    threads:int = 0,
    verbose:int = 0,
    require_chunking:bool = False,
    voxel_graph_fn:Optional[Callable[[int,int], npt.NDArray[np.uint32]]] = None,
) -> dict[int, Skeleton]:
    """
    Extract completed skeletons from a thinned crackle compressed
    image.

    memory: -1, allow use of system memory, otherwise, try to target
        this size in bytes.
    threads: number of threads to use for decompression
    verbose:
        0: no printing
        1: progress bar
        2: current chunk
        3: function timings
    require_chunking:
        This is for testing to force skipping the fast path even if there
        is sufficient memory for it.

    voxel_graph_fn: A function taking z_start and z_end that produces a 26-connected
        uint32 voxel connectivity graph (VCG) in the form given by cc3d. The x and y
        dimensions must match the input labels.

        The VCG is used to disambiguate self-touches (autapses) that would otherwise
        appear to be a single contiguous image.

        Using a VCG will increase the memory requirement.

    Returns: { segid: skeleton }
    """
    import crackle
    from crackle import CrackleArray
    import time
    import psutil
    import multiprocessing as mp
    import tqdm

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
    labels = CrackleArray(labels.binary, parallel=threads)
    header = labels.header()
    vcg = None

    sxy = header.sx * header.sy
    slice_memory = header.data_width * sxy
    # Estimate of crackle per-slice decoding memory usage
    decoding_memory = (header.data_width + 4) * slice_memory * threads # ccl + slice 

    system_memory = psutil.virtual_memory().total
    if memory < 0:
        memory = system_memory

    internal_factor = 1.1 # for extraction data structures

    estimated_memory_requirement = header.data_width * header.voxels()
    estimated_memory_requirement *= internal_factor # for algorithm interal data structures
    estimated_memory_requirement += decoding_memory
    estimated_memory_requirement += len(labels)

    if voxel_graph_fn is not None:
        estimated_memory_requirement += 4 * header.voxels()

    estimated_memory_requirement = int(estimated_memory_requirement)

    if not require_chunking and estimated_memory_requirement < memory and estimated_memory_requirement < system_memory:
        if verbose:
            print("Processing all at once.")
            print(f"Estimated Memory Requirement: {estimated_memory_requirement} bytes")
        
        if voxel_graph_fn is not None:
            vcg = voxel_graph_fn(0, header.sz)

        return extract_skeletons(labels.numpy(), vcg)

    estimated_memory_requirement = decoding_memory + len(labels)

    cz_slice_memory = slice_memory
    if voxel_graph_fn is not None:
        cz_slice_memory += 4 * sxy

    cz = ((memory - estimated_memory_requirement) / internal_factor / cz_slice_memory)
    cz = max(cz, 1)
    cz = min(cz, header.sz)
    cz = int(cz)

    if verbose:
        print(f"chunk size: {cz}")

    estimated_memory_requirement += internal_factor * min(cz, header.sz) * slice_memory

    if voxel_graph_fn is not None:
        estimated_memory_requirement += min(cz, header.sz) * 4 * header.sx * header.sy

    estimated_memory_requirement = int(estimated_memory_requirement)

    if estimated_memory_requirement > min(memory, system_memory):
        raise MemoryError(
            f"Not enough memory to process this volume, try increasing the limit or reducing the number of threads.\n"
            f"User Limit: {memory} bytes\n"
            f"Available: {system_memory} bytes\n"
            f"Estimated memory requirement: {estimated_memory_requirement / 1e9:.1f} GB."
        )

    skeletons = {}
    
    num_chunks = int(np.ceil(header.sz / cz))

    loop_elapsed_sec = 0
    total_elapsed_sec = 0

    for i in tqdm.trange(num_chunks, desc="Slabs", disable=(verbose != 1)):
        s_loop = time.perf_counter()
        z_start = (i * cz) - 1 # need 1 voxel of overlap
        z_start = max(z_start, 0)
        z_end = min((i+1) * cz, header.sz)
        
        percent = f"{i/num_chunks*100.0:.2f}%"
        eta = (num_chunks - i - 1) * loop_elapsed_sec

        if verbose > 1:
            print(f"z={z_start}:{z_end} ; {percent} ; eta {eta:.1f} sec ; elapsed {total_elapsed_sec:.1f} sec")

        s = time.perf_counter()
        arr = labels[:,:, z_start:z_end ]
        e = time.perf_counter()
        if verbose > 2:        
            print(f"decompress: {e - s:.3f}s, {arr.size / (e-s) / 1e6:.1f} MVx/sec")

        if voxel_graph_fn is not None:
            s = time.perf_counter()
            vcg = voxel_graph_fn(z_start, z_end)
            e = time.perf_counter()
            if verbose > 2:        
                print(f"vcg: {e - s:.3f}s, {vcg.size / (e-s) / 1e6:.1f} MVx/sec")

        s = time.perf_counter()
        arr_skeletons = extract_skeletons(arr, vcg)
        e = time.perf_counter()
        if verbose > 2:        
            print(f"extract: {e - s:.3f}s, {arr.size / (e-s) / 1e6:.1f} MVx/sec")
        del arr
        vcg = None

        s = time.perf_counter()
        for segid in arr_skeletons.keys():
            arr_skeletons[segid].vertices[:,2] += z_start

            if segid not in skeletons:
                skeletons[segid] = arr_skeletons[segid]
                continue

            skeletons[segid] = Skeleton.simple_merge([ 
                skeletons[segid], arr_skeletons[segid] 
            ])
        e = time.perf_counter()
        if verbose > 2:        
            print(f"merge: {e - s:.3f}s, {len(skeletons)} skels")

        e_loop = time.perf_counter()
        loop_elapsed_sec = e_loop - s_loop
        total_elapsed_sec += loop_elapsed_sec

    for segid in skeletons:
        skel = skeletons[segid]
        skeletons[segid] = skel.consolidate()

    return skeletons

def skeletonize_crackle(
    labels:Union[CrackleArray, bytes],
    binary_image:bool = True,
    memory:int = -1,
    preserve_endpoints:bool = False,
    anchors:npt.NDArray[np.uint16] = np.empty((0, 3), dtype=np.uint16),
    radius:bool = True,
    anisotropy:tuple[float,float,float] = (1.0, 1.0, 1.0),
    threads:int = 0,
    verbose:int = 0,
    require_chunking:bool = False,
    voxel_graph_fn:Optional[Callable[[int,int], npt.NDArray[np.uint32]]] = None,
) -> tuple[Skeleton, CrackleArray]:

    if isinstance(labels, bytes):
        labels = CrackleArray(labels)
    elif not isinstance(labels, CrackleArray):
        raise ValueError("This function only accepts type CrackleArrays.")

    if not binary_image:
        raise NotImplementedError("multilabel is not implemented yet.")

    labels = labels > 0

    thinned = thin_crackle(
        labels,
        binary_image=True,
        preserve_endpoints=preserve_endpoints,
        memory=memory,
        threads=threads,
        verbose=verbose,
        require_chunking=require_chunking,
        voxel_graph_fn=voxel_graph_fn,
    )

    skeletons = extract_skeletons_crackle(
        thinned,
        memory=memory,
        verbose=verbose,
        require_chunking=require_chunking,
        voxel_graph_fn=voxel_graph_fn,
    )

    if 1 not in skeletons:
        return Skeleton()

    skeleton = skeletons[1]
    anisotropy = np.asarray(anisotropy, dtype=np.float32)
    skeleton.vertices *= anisotropy

    if radius:
        from pykdtree.kdtree import KDTree
        ptc = labels.point_cloud()[1]
        ptc = ptc.astype(np.float32)
        ptc[:] *= anisotropy
        tree = KDTree(ptc)
        radii, _ = tree.query(skeleton.vertices, k=1)
        # regular edt is distance to bg, kdtree returns
        # distance to contour, which is a difference of 1 
        radii += np.min(anisotropy)
        skeleton.add_vertex_attribute("radius", radii)

    return skeleton


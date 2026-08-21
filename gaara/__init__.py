from typing import Union

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
        fastgaara.thin_binary(labels, preserve_endpoints)
    else:
        fastgaara.thin_multilabel(labels, preserve_endpoints)

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
    



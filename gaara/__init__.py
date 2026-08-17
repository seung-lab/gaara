import numpy as np
import numpy.typing as npt

from . import fastgaara

__all__ = ["thin_palagyi"]

def thin_palagyi(labels:npt.NDArray[np.uint8], in_place:bool = False) -> npt.NDArray[np.uint8]:
	"""
	Apply Palagyi's 3D voxel thinning algorithm to `labels`, a binary image.

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

	if in_place and not labels.flags.f_contiguous:
		raise ValueError("Cannot perform an in-place operation on non-Fortran ordered data.")
	elif not in_place:
		labels = np.copy(labels, order="F")

	fastgaara.thin_palagyi(labels)
	return labels









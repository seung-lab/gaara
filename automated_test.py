import numpy as np
import gaara
import pytest

@pytest.mark.parametrize("binary_image", [True, False])
def test_thin(binary_image):
	ones = np.ones([100,100,100], dtype=np.uint64, order="F")
	ones[0,0,0] = 0
	
	gaara.thin(ones, in_place=False, binary_image=binary_image)

	assert np.sum(ones) == 100 * 100 * 100 - 1

	gaara.thin(ones, in_place=True, binary_image=binary_image)

	assert np.sum(ones) == 1

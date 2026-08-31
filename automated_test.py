import numpy as np
import gaara
import pytest
import crackle

@pytest.mark.parametrize("binary_image", [True, False])
def test_thin(binary_image):
	ones = np.ones([3,3,3], dtype=np.uint64, order="F")
	ones[0,0,0] = 0
	
	gaara.thin(ones, in_place=False, binary_image=binary_image)

	assert np.sum(ones) == 3 * 3 * 3 - 1

	gaara.thin(ones, in_place=True, binary_image=binary_image)

	assert np.sum(ones) == 1


def test_extract_skeleton_crackle():
	sz = 200
	img = np.zeros([2000,2000,200], dtype=np.uint64, order="F")
	img[5,5,:] = 1
	img[9,9,:] = 2

	img = crackle.compressa(img)
	skels = gaara.extract_skeletons_crackle(
		img,
		memory=int(1e9),
		require_chunking=True,
		threads=1,
	)

	skel = skels[1]
	assert np.all(skel.vertices[:,0] == 5)
	assert np.all(skel.vertices[:,1] == 5)
	assert np.all(skel.vertices[:,2] == np.arange(200))
	

	skel = skels[2]
	assert np.all(skel.vertices[:,0] == 9)
	assert np.all(skel.vertices[:,1] == 9)
	assert np.all(skel.vertices[:,2] == np.arange(200))
	





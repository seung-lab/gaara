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

@pytest.mark.parametrize("binary_image", [True, False])
def test_skeletonize(binary_image):
	ones = np.zeros([10,10,10], dtype=np.uint64, order="F")
	ones[4:7,4:7,:] = 1
	
	# import microviewer
	# microviewer.view(ones, seg=True)

	skeletons, labels = gaara.skeletonize(
		ones,
		binary_image=binary_image,
		radius=True,
	)

	if not binary_image:
		skeletons = skeletons[1]

	gt = np.array([
		[5, 5, 1],
		[5, 5, 2],
		[5, 5, 3],
		[5, 5, 4],
		[5, 5, 5],
		[5, 5, 6],
		[5, 5, 7],
		[5, 5, 8],
	], dtype=np.uint32)

	assert np.all(gt == skeletons.vertices)
	assert np.all(skeletons.radius == 2)

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
	

def test_skeletonize_vcg_x():
	import cc3d

	sz = 200
	img = np.zeros([200,200,200], dtype=np.uint8, order="F")	

	img[20:-20,80:120,20:-20] = 1
	img[50:-50,:,50:-50] = 0

	skel, thinned = gaara.skeletonize(img, binary_image=True)
	assert len(skel.components()) == 1

	vcg = cc3d.voxel_connectivity_graph(img)

	skel, thinned = gaara.skeletonize(
		img.copy(),
		binary_image=True,
		voxel_graph=vcg
	)
	assert len(skel.components()) == 1

	lhs = 0b1 | (1 << 6) | (1 << 8) | (1 << 10) | (1 << 14) | (1 << 18) | (1 << 20) | (1 << 22) | (1 << 24)
	lhs = ~np.uint32(lhs)

	rhs = 0b10 | (1 << 7) | (1 << 9) | (1 << 11) | (1 << 15) | (1 << 19) | (1 << 21) | (1 << 23) | (1 << 25)
	rhs = ~np.uint32(rhs)

	vcg[100,:,:] = vcg[100,:,:] & lhs
	vcg[101,:,:] = vcg[101,:,:] & rhs

	skel, thinned = gaara.skeletonize(
		img.copy(),
		binary_image=True,
		voxel_graph=vcg
	)

	assert len(skel.components()) == 2

def test_skeletonize_vcg_y():
    import cc3d

    img = np.zeros([200, 200, 200], dtype=np.uint8, order="F")

    img[20:-20, 80:120, 20:-20] = 1
    img[50:-50, :, 50:-50] = 0

    vcg = cc3d.voxel_connectivity_graph(img)

    # keep only +y-directed edges on y=100
    lhs = ~np.uint32(
        0b1 << 2              # +y
      | 0b1 << 6              # xy   (+x+y)
      | 0b1 << 7              # -xy  (-x+y)
      | 0b1 << 12             # yz   (+y+z)
      | 0b1 << 16             # y-z  (+y-z)
      | 0b1 << 18             # xyz  (+x+y+z)
      | 0b1 << 19             # -x+y+z
      | 0b1 << 22             # +x+y-z
      | 0b1 << 23             # -x+y-z
    )

    # keep only -y-directed edges on y=101
    rhs = ~np.uint32(
        0b1 << 3              # -y
      | 0b1 << 8              # x-y  (+x-y)
      | 0b1 << 9              # -x-y
      | 0b1 << 13             # -yz  (-y+z)
      | 0b1 << 17             # -y-z
      | 0b1 << 20             # +x-y+z
      | 0b1 << 21             # -x-y+z
      | 0b1 << 24             # +x-y-z
      | 0b1 << 25             # -x-y-z
    )

    vcg[:,100,:] = vcg[:,100,:] & lhs
    vcg[:,101,:] = vcg[:,101,:] & rhs

    skel, thinned = gaara.skeletonize(
        img.copy(),
        binary_image=True,
        voxel_graph=vcg,
    )
    assert len(skel.components()) == 2

def test_skeletonize_vcg_z():
    import cc3d

    img = np.zeros([200, 200, 200], dtype=np.uint8, order="F")

    img[20:-20, 80:120, 20:-20] = 1
    img[50:-50, :, 50:-50] = 0

    vcg = cc3d.voxel_connectivity_graph(img)

    # keep only +z-directed edges on z=100
    lhs = ~np.uint32(
        0b1 << 4              # +z
      | 0b1 << 10             # xz   (+x+z)
      | 0b1 << 11             # -xz  (-x+z)
      | 0b1 << 12             # yz   (+y+z)
      | 0b1 << 13             # -yz  (-y+z)
      | 0b1 << 18             # xyz  (+x+y+z)
      | 0b1 << 19             # -x+y+z
      | 0b1 << 20             # +x-y+z
      | 0b1 << 21             # -x-y+z
    )

    # keep only -z-directed edges on z=101
    rhs = ~np.uint32(
        0b1 << 5              # -z
      | 0b1 << 14             # x-z  (+x-z)
      | 0b1 << 15             # -x-z
      | 0b1 << 16             # y-z  (+y-z)
      | 0b1 << 17             # -y-z
      | 0b1 << 22             # +x+y-z
      | 0b1 << 23             # -x+y-z
      | 0b1 << 24             # +x-y-z
      | 0b1 << 25             # -x-y-z
    )

    vcg[:, :, 100] = vcg[:, :, 100] & lhs
    vcg[:, :, 101] = vcg[:, :, 101] & rhs

    skel, thinned = gaara.skeletonize(
        img.copy(),
        binary_image=True,
        voxel_graph=vcg,
    )
    assert len(skel.components()) == 2


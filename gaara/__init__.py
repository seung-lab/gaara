import fastgaara
import numpy as np
import numpy.types as npt

def thin_palagyi(labels:npt.NDArray[np.uint8], in_place:bool = True):
	labels = np.asfortranarray(labels)
	fastgaara.









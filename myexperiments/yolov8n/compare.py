import numpy as np

def compare_npy_files(file1, file2):
	array1 = np.load(file1)
	array2 = np.load(file2)

	if np.array_equal(array1, array2):
		print("two arrays are identical")
	else:
		print("different")
		diff = np.abs(array1 - array2)
		print("\narray1:")
		print(array1)
  
		print("\narray2:")
		print(array2)
  
		print("\nDifference between arrays:")
		print(diff)
  
		print("\nmax difference : ", np.max(diff))
		print(  "avg difference : ", np.mean(diff))

	
file1 = "omout.npy"
file2 = "ortout.npy"

compare_npy_files(file1, file2)

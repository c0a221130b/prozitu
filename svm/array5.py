import numpy as np

nums = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [-1, -2]]], np.int32)

print(nums[2:4, :, 1])

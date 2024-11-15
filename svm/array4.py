import numpy as np

nums = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int32)

print(nums[:, 1])

print(nums[1:3, 1])
print(nums[:1, 2])
print(nums[2:3, 0])
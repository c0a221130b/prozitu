#!/home/hina/anaconda3/bin/python
import re
import os
import numpy as np
import cv2
img_gray = np.zeros((1, 3), np.uint8)
with open(f"./num2/num2_1", mode="rb") as f:
    b = f.read()
res1 = re.search(b"\x00\x00\x8b\x45", b)
res2 = re.search(b"\x45\xfc\x03\x00", b)
res3 = re.search(b"\x89\x45\xf8\xb8", b)
if res1:
 img_gray[0, 0] = 255

if res2:
 img_gray[0, 1] = 255

if res3:
 img_gray[0, 2] = 255
filename = f"/home/hina/mouse/learning/vector.png"
cv2.imwrite(f'{filename}', img_gray)
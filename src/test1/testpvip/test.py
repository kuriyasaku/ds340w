from PIL import Image
import numpy as np

img1 = np.array(Image.open("pyvips.png"))
img2 = np.array(Image.open("pillow.png"))

print(img1.shape, img2.shape)
print(np.array_equal(img1, img2))

if img1.shape == img2.shape:
    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
    print("max diff:", diff.max())
    print("mean diff:", diff.mean())
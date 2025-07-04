import math
import numpy as np

"""
# the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images
"""
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= len(imageA)

    return err

def psnr2(img1, img2, valid):
    mae = np.sum(img1 - img2) / valid
    mse = (np.sum((img1 - img2) ** 2)) / valid
    if mse < 1.0e-10:
      return 100
    PIXEL_MAX = 1
    # print(mse)
    return mae, mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
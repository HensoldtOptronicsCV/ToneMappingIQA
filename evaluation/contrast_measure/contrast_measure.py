# MIT License
#
# Copyright (c) 2020 HENSOLDT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# *
# Python script to calculate the contrast measure provided by Eilertsen et al.
# Link to paper: https://www.cl.cam.ac.uk/~rkm38/pdfs/eilertsen2017tmo_star.pdf
# * #

# todo: remove unnecessary imports
import cv2
import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from skimage import restoration
from skimage.restoration import denoise_bilateral
import OpenEXR
import array
import Imath
from PIL import Image
import struct

# todo: add procedure that iterates through all frames in a given directory path

def preprocessImage(image, normValue):
    # normalize
    imageNorm = np.float64(image) / normValue
    # remove negative and zero values of the tone mapped image so that log transform can be performed
    imageNorm[imageNorm <= 0] = min(imageNorm[imageNorm > 0])
    imageNormLog = np.log10(imageNorm**(2.2)) # denoise_bilateral only works for positiv values
    
    return np.float32(imageNormLog)


def calcContrastMeasure(imageHDR, imageLDR):
    # calculate the contrast measure for one given pair of frames (tone mapped image together with corresponding HDR image)
    
    imageLDR_32 = preprocessImage(imageLDR, 2**8 - 1)
    imageHDR_32 = preprocessImage(imageHDR, 2**14 - 1)

    # calculate dimension of the tone mapped image
    height, width = imageLDR.shape
    # calculate size of the tone mapped image for the variable N in Eq. 28 from paper
    size = imageLDR.size

    # calculate local contrast for the LDR image (Eq. 27 from paper)
    c_local1_T = cv2.bilateralFilter(imageLDR_32, -1, 0.2, 10)
    c_local2_T = imageLDR_32 - c_local1_T
    c_local_T = np.mean(imageLDR_32*np.abs(c_local2_T))

    # calculate local contrast for the HDR image (Eq. 27 from paper)
    c_local1_L = cv2.bilateralFilter(imageHDR_32, -1, 0.2, 10)
    c_local2_L = imageHDR_32 - c_local1_L
    c_local_L = np.mean(imageHDR_32 * np.abs(c_local2_L))
    
    # calculate result: loss of local contrast
    loss_local_contrast = c_local_T - c_local_L

    # calculate global contrast for the tone mapped image (Eq. 28 from paper)
    img_gaussian1_global_T = cv2.GaussianBlur(imageLDR_32, (9, 9), 3)
    img_gaussian1_global_T = img_gaussian1_global_T*img_gaussian1_global_T
    img_gaussian2_global_T = cv2.GaussianBlur(imageLDR_32*imageLDR_32, (9, 9), 3)
    img_gaussian_global_T = np.array(np.sqrt(np.abs(img_gaussian2_global_T-img_gaussian1_global_T)))

    c_global_T = 0
    for y in range(height):
        for x in range(width):
            c_global_T = c_global_T - img_gaussian_global_T[y, x]
    c_global_T /= size

    # calculate global contrast for the HDR image (Eq. 28 from paper)
    img_gaussian1_global_L = cv2.GaussianBlur(imageHDR_32, (9, 9), 3)
    img_gaussian1_global_L = img_gaussian1_global_L*img_gaussian1_global_L
    img_gaussian2_global_L = cv2.GaussianBlur(imageHDR_32*imageHDR_32, (9, 9), 3)
    img_gaussian_global_L = np.array(np.sqrt(np.abs(img_gaussian2_global_L-img_gaussian1_global_L)))

    c_global_L = 0
    for y in range(height):
        for x in range(width):
            c_global_L = c_global_L - img_gaussian_global_L[y, x]
    c_global_L /= size
    
    # calculate result: loss of global contrast
    loss_global_contrast = c_global_T - c_global_L
    
    return loss_local_contrast, loss_global_contrast


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
# Python script to calculate the exposure measure provided by Eilertsen et al.
# Link to paper: https://www.cl.cam.ac.uk/~rkm38/pdfs/eilertsen2017tmo_star.pdf
# * #

import numpy as np
import cv2


def calcExposureMeasure(image):
    # calculate exposure measure for one given frame
    
    # calculate histogram of the image
    hist, bins = np.histogram(image, 256, [0, 255])
    # fraction of under-exposed pixels
    under_exp_pix = sum(hist[0:int(255*0.02)])/sum(hist) * 100
    # fraction of over-exposed pixels
    over_exp_pix = sum(hist[int(255*0.95):])/sum(hist) * 100
    
    return under_exp_pix, over_exp_pix


def exposure(pathToToneMappedImages):
    # calcuate exposure measure for all (already tone mapped!) frames in given path
    
    sequence_length = len(pathToToneMappedImages)
    under_exp_pix = 0
    over_exp_pix = 0

    for image in pathToToneMappedImages:
        # read tone mapped (TM) image as grayscale image
        toneMappedImage = cv2.imread(str(image), 0)
        
        curr_under_exp_pix, curr_over_exp_pix = calcExposureMeasure(toneMappedImage)

        # fraction of under-exposed pixels
        under_exp_pix += curr_under_exp_pix

        # fraction of over-exposed pixels
        over_exp_pix += curr_over_exp_pix

    # calculate average of under- and over-exposed pixels
    under_exp = under_exp_pix/sequence_length
    over_exp = over_exp_pix/sequence_length

    return under_exp, over_exp



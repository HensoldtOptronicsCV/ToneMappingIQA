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

# Python implementation of the under/over-exposure measure. We focus on simplicity and readability rather than efficiency.
#
# This code is related to the paper 
# M. Teutsch, S. Sedelmaier, S. Moosbauer, G. Eilertsen, T. Walter, 
# "An Evaluation of Objective Image Quality Assessment for Thermal Infrared Video Tone Mapping", IEEE CVPR Workshops, 2020.
#
# Please cite the paper if you use the code for your evaluations.

# This measure was originally proposed here:
# G. Eilertsen, R. Mantiuk, J. Unger, "A comparative review of tone-mapping algorithms for high dynamic range video", Eurographics, 2017.

import numpy as np
import cv2

## Calcuate the over- and under-exposure measure (number of over- and under-exposed pixels) for one given tone mapped LDR image.
#  @param image_ldr Low Definition Range image (processed image after tone mapping).
def number_of_over_and_under_exposures_pixels(image_ldr):
    # calculate exposure measure for one given frame
    
    # calculate histogram of the image
    hist, bins = np.histogram(image_ldr, 256, [0, 255])
    
    # fraction of under-exposed pixels
    under_exp_pix = sum(hist[0:int(255 * 0.02)])/sum(hist) * 100
    
    # fraction of over-exposed pixels
    over_exp_pix = sum(hist[int(255 * 0.95):])/sum(hist) * 100
    
    return over_exp_pix, under_exp_pix


## Calculate over- and under-exposure measure for all (already tone mapped) images in given path.
#  @param images_ldr_path Directory path that contains the tone mapped images of one sequence.
def calculate_over_and_under_exposure_measure(images_ldr_path):
    
    sequence_length = len(images_ldr_path)
    
    if sequence_length == 0:
        raise ValueError('List of LDR image paths must not be empty.')
    
    under_exp_pix = 0
    over_exp_pix = 0

    for image_ldr_path in images_ldr_path:
        print(".", end = '', flush = True) # show progress
        # read tone mapped (TM) image as grayscale image
        image_ldr = cv2.imread(str(image_ldr_path), cv2.IMREAD_GRAYSCALE)
        
        curr_over_exp_pix, curr_under_exp_pix = number_of_over_and_under_exposures_pixels(image_ldr)

        # fraction of under-exposed pixels
        under_exp_pix += curr_under_exp_pix

        # fraction of over-exposed pixels
        over_exp_pix += curr_over_exp_pix

    # calculate average of over- and under-exposed pixels for this sequence
    over_exposure = over_exp_pix / sequence_length
    under_exposure = under_exp_pix / sequence_length
    
    print() # newline after progress dots
    
    return over_exposure, under_exposure



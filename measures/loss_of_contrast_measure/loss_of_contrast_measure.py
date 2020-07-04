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

# Python implementation of the loss of contrast measure. We focus on simplicity and readability rather than efficiency.
#
# This code is related to the paper 
# M. Teutsch, S. Sedelmaier, S. Moosbauer, G. Eilertsen, T. Walter, 
# "An Evaluation of Objective Image Quality Assessment for Thermal Infrared Video Tone Mapping", IEEE CVPR Workshops, 2020.
#
# Please cite the paper if you use the code for your evaluations.

# This measure was originally proposed here:
# G. Eilertsen, R. Mantiuk, J. Unger, "A comparative review of tone-mapping algorithms for high dynamic range video", Eurographics, 2017.

import cv2
import numpy as np
from utils import preprocess_image

# global variables
gamma_correction = 2.2

## Calculate the global loss of contrast measure for one given pair of frames (tone mapped image together with corresponding HDR image).
#  @param image_hdr High Definition Range image (original image before tone mapping).
#  @param image_ldr Low Definition Range image (related processed image after tone mapping).
#
#  NOTE: The images are expected to be pre-processed already (i.e. normalized and logarithmized).
def measure_global_contrast(image_hdr, image_ldr):
    kernel_size = 9, 9
    sigma = 3
    
    # bring HDR and LDR image to log domain
    image_hdr = np.log10(image_hdr)
    image_ldr = np.log10(image_ldr)

    # calculate global contrast for the tone mapped LDR image (Eq. 4)
    image_gaussian1_global_T = cv2.GaussianBlur(image_ldr, kernel_size, sigma)
    image_gaussian1_global_T *= image_gaussian1_global_T
    image_gaussian2_global_T = cv2.GaussianBlur(image_ldr ** 2, kernel_size, sigma)
    image_gaussian_global_T = np.array(np.sqrt(np.abs(image_gaussian2_global_T - image_gaussian1_global_T)))
    C_global_T = -(np.sum(image_gaussian_global_T) / image_ldr.size)
    
    # calculate global contrast for the HDR image (Eq. 4)
    image_gaussian1_global_L = cv2.GaussianBlur(image_hdr, kernel_size, sigma)
    image_gaussian1_global_L *= image_gaussian1_global_L
    image_gaussian2_global_L = cv2.GaussianBlur(image_hdr ** 2, kernel_size, sigma)
    image_gaussian_global_L = np.array(np.sqrt(np.abs(image_gaussian2_global_L - image_gaussian1_global_L)))
    C_global_L = -(np.sum(image_gaussian_global_L) / image_ldr.size)
        
    # calculate result: loss of global contrast
    loss_global_contrast = C_global_T - C_global_L
    
    return loss_global_contrast


## Calculate the local loss of contrast measure for one given pair of frames (tone mapped image together with corresponding HDR image).
#  @param image_hdr High Definition Range image (original image before tone mapping).
#  @param image_ldr Low Definition Range image (related processed image after tone mapping).
#
#  NOTE: The images are expected to be pre-processed already (i.e. normalized and logarithmized).
def measure_local_contrast(image_hdr, image_ldr):
    sigma_color = 0.2
    sigma_space = 10
    
    # bring HDR and LDR image to log domain
    image_hdr = np.log10(image_hdr)
    image_ldr = np.log10(image_ldr)
     
    # calculate local contrast for the tone mapped LDR image (Eqs. 2 and 3)
    # the parameter '-1' makes the function compute the filter size using sigma_space
    C_local1_T = cv2.bilateralFilter(image_ldr, -1, sigma_color, sigma_space)
    C_local2_T = image_ldr - C_local1_T
    C_local_T = np.mean(image_ldr * np.abs(C_local2_T))

    # calculate local contrast for the HDR image (Eqs. 2 and 3)
    # the parameter '-1' makes the function compute the filter size using sigma_space
    C_local1_L = cv2.bilateralFilter(image_hdr, -1, sigma_color, sigma_space)
    C_local2_L = image_hdr - C_local1_L
    C_local_L = np.mean(image_hdr * np.abs(C_local2_L))
    
    # calculate result: loss of local contrast
    loss_local_contrast = C_local_T - C_local_L
    
    return loss_local_contrast


## Calculate global and local loss of contrast measure for all HDR and LDR image pairs in given path.
#  @param images_hdr_path Contains the absolute directory paths to the original HDR images of one sequence.
#  @param images_ldr_path Contains the absolute directory paths to the related tone mapped LDR images of one sequence.
#  @param norm_value_hdr Normalization value for HDR images.
#  @param norm_value_ldr Normalization value for LDR images.
def calculate_loss_of_contrast(images_hdr_path, images_ldr_path, norm_value_hdr, norm_value_ldr):
        
    if len(images_hdr_path) != len(images_ldr_path):
        raise ValueError('Sizes of HDR and LDR image lists are different.')
    
    if len(images_hdr_path) == 0:
        raise ValueError('List of HDR or LDR image paths must not be empty.')
    
    loss_global_contrast = 0
    loss_local_contrast = 0
    for hdr_image_path, ldr_image_path in zip(images_hdr_path, images_ldr_path):
        print(".", end = '', flush = True) # show progress
        image_hdr = preprocess_image(cv2.imread(str(hdr_image_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH), norm_value_hdr)
        image_ldr = preprocess_image(cv2.imread(str(ldr_image_path), cv2.IMREAD_GRAYSCALE), norm_value_ldr, gamma_correction)
        loss_global_contrast += measure_global_contrast(image_hdr, image_ldr)
        loss_local_contrast += measure_local_contrast(image_hdr, image_ldr)
    
    loss_global_contrast /= len(images_hdr_path)
    loss_local_contrast /= len(images_hdr_path)
    
    print() # newline after progress dots

    return loss_global_contrast, loss_local_contrast


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

# Python implementation of the temporal incoherence measure. We focus on simplicity and readability
# rather than efficiency.
#
# This code is related to the paper 
# M. Teutsch, S. Sedelmaier, S. Moosbauer, G. Eilertsen, T. Walter, 
# "An Evaluation of Objective Image Quality Assessment for Thermal Infrared Video Tone Mapping",
# IEEE CVPR Workshops, 2020.
#
# Please cite the paper if you use the code for your evaluations.

# The temporal incoherence measure was originally proposed here:
# G. Eilertsen, R. Mantiuk, J. Unger, "A comparative review of tone-mapping algorithms for high dynamic range video",
# Eurographics, 2017.

import sys
import cv2
import numpy as np
import numpy.matlib as np_mat
from utils import *

# global variables
gamma_correction = 2.2
upper_mask_threshold = 1.0 - 1.0 / 255.0 # taken from the original paper
lower_mask_threshold = 0.2 # taken from the original paper
slope = 0.25 # taken from the original paper
temporal_radius = 5  # number of images per bunch: 2*temporal_radius + 1

# Calculate the global temporal incoherence measure for one given set of image pairs (tone mapped image together
# with corresponding HDR image).
#  @param images_hdr High Definition Range images (original image before tone mapping).
#  @param images_ldr Low Definition Range images (related processed image after tone mapping).
#
#  NOTE: The images are expected to be pre-processed already (i.e. normalized and logarithmized).
def measure_global_temporal_incoherence( images_hdr, images_ldr):
    if len(images_hdr) != len(images_ldr):
        raise ValueError('Number of elements in HDR and LDR image arrays are not similar.')

    if images_hdr[0].shape != images_ldr[0].shape:
        raise ValueError('Images dimensions of HDR and LDR images do not match.')

    d = temporal_radius

    # bring HDR and LDR image to log domain
    images_hdr = np.log10(images_hdr)
    images_ldr = np.log10(images_ldr)

    # calculate image means
    means_hdr = []
    for image in images_hdr:
        curr_mean = cv2.mean(image)
        # OpenCV calculates the mean channel-wise, so we must actively access the intensity (=first) channel
        means_hdr.append(curr_mean[0])
    means_ldr = []
    for image in images_ldr:
        curr_mean = cv2.mean(image)
        # OpenCV calculates the mean channel-wise, so we must actively access the intensity (=first) channel
        means_ldr.append(curr_mean[0])

    # indices X of elements with index 0 in the center (Eq. 12)
    X = np.array(range(-d, d + 1))  # d is temporal radius

    if len(images_hdr) != len(X):
        raise ValueError(
            'Number of HDR images {} does not match the expected temporal radius of {}.'.format(len(images_hdr),
                len(X)))

    # linear regression (Eqs. 13 and 14)
    w_L = np.sum(np.multiply(means_hdr, X)) / np.dot(X, X)
    w_T = np.sum(np.multiply(means_ldr, X)) / np.dot(X, X)

    # temporal averaging kernel K_i (Eq. 15)
    K_i = np.ones(2 * d + 1) / (2 * d + 1)

    # fitted lines y_L for HDR image and y_T for tone mapped LDR image (Eqs. 16 and 17)
    y_L = w_L * X + np.sum(np.multiply(K_i, means_hdr))
    y_T = w_T * X + np.sum(np.multiply(K_i, means_ldr))

    # normalize with the fitted lines (Eqs. 18 and 19)
    t_hat_L = np.subtract(means_hdr, y_L)
    t_hat_T = np.subtract(means_ldr, y_T)

    # variances (Eqs. 20 and 21)
    var_L = np.sum(np.multiply(K_i, np.square(t_hat_L)))
    var_T = np.sum(np.multiply(K_i, np.square(t_hat_T)))

    # variance normalization of signals (Eq. 22)
    if np.sqrt(var_L) < np.finfo(np.float32).eps:  # avoid division-by-zero
        var_L = 0.00001
    t_L_norm = np.multiply(t_hat_L, np.sqrt(np.divide(var_T, var_L)))
    t_T_norm = t_hat_T

    # put back the linear slope of signal L (Eqs. 23 and 24)
    # now signal T has same slope and variance as signal L
    t_L = slope * X + t_L_norm
    t_T = slope * X + t_T_norm

    # pearson correlation coefficient (Eqs. 25, 26, 27)
    q1 = np.sum(np.multiply(K_i, np.square(t_L)))
    q2 = np.sum(np.multiply(K_i, np.square(t_T)))
    q3 = np.sum(np.multiply(K_i, np.multiply(t_L, t_T)))

    # global temporal incoherence measure (Eq. 28)
    cf = 1.0 - max(0, q3 / np.sqrt(q1 * q2))

    return cf


# Calculate the local temporal incoherence measure for one given set of image pairs
# (tone mapped image together with corresponding HDR image).
#  @param images_hdr High Definition Range images (original image before tone mapping).
#  @param images_ldr Low Definition Range images (related processed image after tone mapping).
#
#  NOTE: The images are expected to be pre-processed already (i.e. normalized and logarithmized).
def measure_local_temporal_incoherence( images_hdr, images_ldr, mask):
    if len(images_hdr) != len(images_ldr):
        raise ValueError('number of elements in HDR and LDR image arrays are not similar')

    if images_hdr[0].shape != images_ldr[0].shape:
        raise ValueError('images dimensions of HDR and LDR images do not match')

    # bring HDR and LDR image to log domain
    images_hdr_log = np.log10(images_hdr)
    images_ldr_log = np.log10(images_ldr)

    d = temporal_radius

    # indices of elements with index 0 in the center (Eq. 12)
    X = np.array(range(-d, d + 1))

    if len(images_hdr_log) != len(X):
        raise ValueError(
            'Number of HDR images {} does not match the expected temporal radius of {}.'.format(len(images_hdr_log), len(X))
        )

    # pixelwise linear regression (Eqs. 13 and 14)
    w_L = np.zeros(images_hdr_log[0].shape)
    w_T = np.zeros(images_ldr_log[0].shape)

    norm_factor = np.sum(np.multiply(X, X))
    for i in range(len(images_hdr_log)):
        w_L += np.divide(X[i] * np.array(images_hdr_log[i]), norm_factor)
        w_T += np.divide(X[i] * np.array(images_ldr_log[i]), norm_factor)

    # temporal averaging kernel (Eq. 15)
    K_i = np.ones(2 * d + 1) / (2 * d + 1)
    K_i_reshaped = [np_mat.repmat(1 / (2 * d + 1), images_hdr_log[0].shape[0], images_hdr_log[0].shape[1]) for _ in range(2 * d + 1)]
    K_i_reshaped = np.array(np.stack(K_i_reshaped, axis = 0))

    image_stack = np.array(np.stack(images_hdr_log, axis = 0))
    m_L = np.sum(np.multiply(K_i_reshaped, image_stack), 0)

    image_stack = np.array(np.stack(images_ldr_log, axis = 0))
    m_T = np.sum(np.multiply(K_i_reshaped, image_stack), axis = 0)

    # pixelwise fitted lines
    y_L = [np.add(x * np.array(w_L), m_L) for x in X]
    y_T = [np.add(x * np.array(w_T), m_T) for x in X]

    # normalize with the fitted lines
    t_hat_L = np.subtract(images_hdr_log, y_L)
    t_hat_T = np.subtract(images_ldr_log, y_T)

    # variances
    var_L = np.zeros(images_hdr_log[0].shape)
    var_T = np.zeros(images_ldr_log[0].shape)
    for i, k_i in enumerate(K_i):
        var_L += (k_i * np.array(np.square(t_hat_L[i])))
        var_T += (k_i * np.array(np.square(t_hat_T[i])))

    # variance normalization of signals
    divisor = np.maximum(np.sqrt(var_L), np.full(var_L.shape, np.finfo(np.float32).eps))  # avoid division-by-zero
    t_L_norm = np.multiply(t_hat_L, np.divide(np.sqrt(var_T), divisor))
    t_T_norm = t_hat_T

    # put back the linear slope of signal L
    # now signal T has same slope and variance as signal L
    t_L = []
    t_T = []
    for i, x in enumerate(X):
        t_L.append(x * np.full(images_hdr_log[0].shape, slope) + t_L_norm[i])
        t_T.append(x * np.full(images_ldr_log[0].shape, slope) + t_T_norm[i])

    # pixelwise pearson correlation coefficient
    t_L_stacked = np.stack(t_L, axis = 0)
    t_T_stacked = np.stack(t_T, axis = 0)
    q1 = np.sum(np.multiply(K_i_reshaped, np.square(t_L_stacked)), axis = 0)
    q2 = np.sum(np.multiply(K_i_reshaped, np.square(t_T_stacked)), axis = 0)
    q3 = np.sum(np.multiply(K_i_reshaped, np.multiply(t_L_stacked, t_T_stacked)), axis = 0)

    # determine pixelwise local temporal coherence measure
    # avoid division-by-zero
    divisor = np.maximum(np.sqrt(np.multiply(q1, q2)), np.full(q1.shape, np.finfo(np.float32).eps))
    cf = np.maximum(0, np.divide(q3, divisor))

    # invert value to get pixelwise local temporal incoherence measure
    # we need the central HDR image here in original and not in log domain
    cf = np.where((images_hdr[d] >= 1e-5), cf, 1.0)

    cf = 1.0 - cf

    # weight local incoherence measure based on the pixelwise variance
    # we need the LDR images here in original and not in log domain
    W = np.multiply(np.sum(np.multiply(K_i_reshaped, np.sqrt(images_ldr)), axis = 0), var_T)
    W /= np.mean(np.sqrt(images_ldr)) * np.mean(var_T)
    cf = np.multiply(cf, W)

    # calculate local pixelwise measure only on values which are not under- or over-exposed
    binary_mask = np.where(mask, 1, 0)
    masked_stack = np.multiply(cf, binary_mask)

    cf_local2 = np.sum(np.where(masked_stack > 0.05, masked_stack, 0))
    cf_local2 /= np.sum(binary_mask)

    return cf_local2


# Calculate global and local temporal incoherence measure for all HDR and LDR image pairs in given path.
#  @param images_hdr_path Directory path that contains the original HDR images of one sequence.
#  @param images_ldr_path Directory path that contains the related tone mapped LDR images of one sequence.
#  @param norm_value_hdr Normalization value for HDR images.
#  @param norm_value_ldr Normalization value for LDR images.
def calculate_temporal_incoherence_measure( images_hdr_path, images_ldr_path, norm_value_hdr, norm_value_ldr ):
    sequence_length = len(images_hdr_path)

    if sequence_length == 0:
        raise ValueError('List of HDR or LDR image paths must not be empty.')

    if sequence_length < 2 * temporal_radius + 1:
        raise ValueError('Number of images in given sequence is too small to calculate temporal incoherence measure.')

    if sequence_length != len(images_ldr_path):
        raise ValueError('Sizes of HDR and LDR image lists are different.')

    global_temporal_incoherence = 0
    local_temporal_incoherence = 0
    # we have to iterate starting from a temporal offset given by temporal_radius
    for i in range(temporal_radius, sequence_length - temporal_radius):
        print(".", end = '', flush = True) # show progress
        # get bunch of images needed as temporal context for calculating the measure
        if i == temporal_radius:
            images_hdr = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) for image in images_hdr_path[i - temporal_radius: i + temporal_radius + 1]]

            images_ldr = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) for image in images_ldr_path[i - temporal_radius: i + temporal_radius + 1]]
        else:
            images_hdr = images_hdr[1:]
            new_hdr = cv2.imread(str(images_hdr_path[i + temporal_radius]), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

            images_ldr = images_ldr[1:]
            new_ldr = cv2.imread(str(images_ldr_path[i + temporal_radius]), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

            images_hdr.append(new_hdr)
            images_ldr.append(new_ldr)

        assert (len(images_hdr) == 2 * temporal_radius + 1)
        images_hdr_preproc = preprocess_image_list(images_hdr, norm_value_hdr)
        images_ldr_preproc = preprocess_image_list(images_ldr, norm_value_ldr, gamma_correction)
        global_temporal_incoherence += measure_global_temporal_incoherence(images_hdr_preproc, images_ldr_preproc)

        if gamma_correction > 1.0:
            mask = ((lower_mask_threshold < np.power(images_ldr_preproc, 0.5)) &
                    (np.power(images_ldr_preproc, 0.5) < upper_mask_threshold))
        else:
            mask = ((lower_mask_threshold**2 < np.array(images_ldr_preproc)) &
                    (np.array(images_ldr_preproc) < upper_mask_threshold**2))

        local_temporal_incoherence += measure_local_temporal_incoherence(images_hdr_preproc, images_ldr_preproc, mask)

    global_temporal_incoherence /= sequence_length
    local_temporal_incoherence /= sequence_length
    
    print() # newline after progress dots

    return global_temporal_incoherence, local_temporal_incoherence

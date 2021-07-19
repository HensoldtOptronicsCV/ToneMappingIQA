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


# Standard library imports
import os
import argparse

# Third party imports
import numpy as np
import cv2

# Local application imports
from noise_visibility_measure.calculate_noise_visibility import noise_visibility

if __name__ == "__main__":
    # parsing input arguments
    parser = argparse.ArgumentParser(description='Paths for hdr images and tonemapped images')
    parser.add_argument('hdr_ref_image_path', type=str, help='folder path for the reference hdr images')
    parser.add_argument('hdr_noisy_image_path', type=str, help='folder path for the noisy hdr images')
    parser.add_argument('tmo_ref_image_path', type=str, help='folder path for the tonemapped reference images')
    parser.add_argument('tmo_noisy_image_path', type=str, help='folder path for the tonemapped noisy images')
    image_folder_paths = parser.parse_args()

    # lists of the path contents, ordered alphabetically.
    hdr_ref_list = sorted(os.listdir(image_folder_paths.hdr_ref_image_path))
    hdr_noisy_list = sorted(os.listdir(image_folder_paths.hdr_noisy_image_path))
    tmo_ref_list = sorted(os.listdir(image_folder_paths.tmo_ref_image_path))
    tmo_noisy_list = sorted(os.listdir(image_folder_paths.tmo_noisy_image_path))

    # check if number of images is equal in every given directory
    if not len(hdr_ref_list) == len(hdr_noisy_list) == len(tmo_ref_list) == len(tmo_noisy_list):
        raise RuntimeError("Number of images is not the same in every given folder")

    # check if folders are empty
    if len(hdr_ref_list) + len(hdr_noisy_list) + len(tmo_ref_list) + len(tmo_noisy_list) == 0:
        raise RuntimeError("Given folders are empty")

    # printing statements and initializing progress bar
    print("Calculate noise visibility measure")
    num_samples = len(hdr_ref_list)                 # number of needed iterations
    steps = np.clip(num_samples, 1, 100)           # determines how often the Progress bar will be refreshed (roughly)
    iterations_to_print = np.linspace(1, num_samples, round(steps), dtype=int) # iterations where the progress bar will be updated

    # setting variables for loop
    list_n = np.array([])
    counter = 0

    # For loop iterating through images in given directories and calculating noise visibility measure for image pairs
    for hdr_ref, hdr_noisy, tmo_ref, tmo_noisy in zip(hdr_ref_list, hdr_noisy_list, tmo_ref_list, tmo_noisy_list):
        # comparing reference hdr image with noisy hdr image
        image_ref = cv2.imread(str(image_folder_paths.hdr_ref_image_path + "/" + hdr_ref), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        image_noisy = cv2.imread(str(image_folder_paths.hdr_noisy_image_path + "/" + hdr_noisy), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

        hdr_Q = noise_visibility(image_ref, image_noisy)    # q-metric of original hdr image and hdr image with added noise

        # comparing tonemapped reference image with tonemapped noisy image
        image_ref = cv2.imread(str(image_folder_paths.tmo_ref_image_path + "/" + tmo_ref), cv2.IMREAD_GRAYSCALE)
        image_noisy = cv2.imread(str(image_folder_paths.tmo_noisy_image_path + "/" + tmo_noisy), cv2.IMREAD_GRAYSCALE)

        tmo_Q = noise_visibility(image_ref, image_noisy)    # q-metric of tonemapped hdr image and tonemapped noisy hdr image

        # calculating Noise Visibility Measure formula (8) in paper
        n = hdr_Q - tmo_Q
        list_n = np.append(list_n, n)

        counter += 1

        # update Progress bar
        if any(counter == iterations_to_print):
            print('\r', "Progress: {}%".format(round(counter / num_samples * 100 - 0.5)), end="")

    print()
    print("noise visibility measure for {} HDR/noisyHDR image pairs and {} LDR/noisyLDR image pairs: {}".format(counter, counter, sum(list_n)/counter))
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

if __name__ == "__main__":
    # parsing input arguments
    parser = argparse.ArgumentParser(description='adds poisson noise to images from first folder path and saves them in second folder path')
    parser.add_argument('ref_image_path', type=str, help='folder path for the reference images (where noise should be added)')
    parser.add_argument('noisy_image_path', type=str, help='folder path where the noisy images should be saved to')
    args = parser.parse_args()

    # Value for normalization of hdr images. Default for 16 bit hdr images is 65535
    norm_value = 65535  # 65535 for 16 bit images and 255 for 8 bit images
    poisson_lambda = 100

    # check if folder for noisy images exists, if not, folder is created
    if not os.path.exists(args.noisy_image_path):
        os.makedirs(args.noisy_image_path)

    # setting variables for progress bar
    num_samples = len(os.listdir(args.ref_image_path))   # number of needed iterations
    steps = np.clip(num_samples, 1, 100)                         # determines how often the Progress bar will be refreshed (roughly)
    iterations_to_print = np.linspace(1, num_samples, round(steps), dtype=int)  # iterations where the progress bar will be updated
    counter = 0

    # for loop iterating trough images in given folder and saving version with added poisson noise in noisy_image_path
    for image_name in sorted(os.listdir(args.ref_image_path)):
        image = cv2.imread(str(args.ref_image_path + "/" + image_name), cv2.IMREAD_ANYDEPTH)
        row, col = image.shape
        poisson = np.random.poisson(poisson_lambda, (row, col))
        poisson = poisson.reshape(row, col)
        noisy_image = np.clip(image + poisson, 0, norm_value)
        noisy_image = np.array(noisy_image, dtype=np.uint16)
        filename = 'noisy' + image_name
        cv2.imwrite(args.noisy_image_path + '/' + filename, noisy_image)
        counter += 1

        # refresh progress bar
        if any(counter == iterations_to_print):
            print('\r', "Progress: {}%".format(round(counter / num_samples * 100 - 0.5)), end="")
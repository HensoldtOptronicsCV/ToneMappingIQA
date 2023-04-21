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

import json
import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve().parent))

import cv2
import numpy as np
from tqdm import tqdm

from measures.base_measure import BaseMeasure


class OverUnderExposure(BaseMeasure):
    def calculate_over_and_under_exposure_measure(self, seq_name):
        """
        Calculate over- and under-exposure measure for all (already tone mapped) images in given path
        :param seq_name: Name of sequence to calculate the measure for
        :return:
        """
        ldr_images_path = self.experiment_data[seq_name]["ldr"]
        sequence_length = len(ldr_images_path)

        assert sequence_length > 0

        under_exp_pix = 0
        over_exp_pix = 0

        with tqdm(total=sequence_length) as pbar:
            for image_ldr_path in ldr_images_path:
                # read tone mapped (TM) image as grayscale image
                image_ldr = cv2.imread(str(image_ldr_path), cv2.IMREAD_GRAYSCALE)

                curr_over_exp_pix, curr_under_exp_pix = self.number_of_over_and_under_exposures_pixels(image_ldr)

                # fraction of under-exposed pixels
                under_exp_pix += curr_under_exp_pix

                # fraction of over-exposed pixels
                over_exp_pix += curr_over_exp_pix
                pbar.update(1)

            # calculate average of over- and under-exposed pixels for this sequence
            over_exposure = over_exp_pix / sequence_length
            under_exposure = under_exp_pix / sequence_length

        return over_exposure, under_exposure

    def number_of_over_and_under_exposures_pixels(self, ldr_image):
        """
        Calcuate the over- and under-exposure measure (number of over- and under-exposed pixels) for one given tone mapped LDR image
        :param ldr_image: Low Definition Range image (processed image after tone mapping)
        :return:
        """
        # calculate exposure measure for one given frame

        # calculate histogram of the image
        hist, bins = np.histogram(ldr_image, 256, [0, 255])

        # fraction of under-exposed pixels
        under_exp_pix = sum(hist[0:int(255 * 0.02)]) / sum(hist) * 100

        # fraction of over-exposed pixels
        over_exp_pix = sum(hist[int(255 * 0.95):]) / sum(hist) * 100

        return over_exp_pix, under_exp_pix

    def calculate(self):
        over_exposure = 0
        under_exposure = 0
        number_of_images = 0
        for sequence_name in self.eval_sequences:
            if sequence_name not in self.dataset_data:
                raise RuntimeError(f"Evaluation sequence '{sequence_name}' not found in evaluation dataset.")
            number_of_images += len(self.experiment_data[sequence_name]['ldr'])
            curr_over_exposure, curr_under_exposure = self.calculate_over_and_under_exposure_measure(sequence_name)
            over_exposure += curr_over_exposure
            under_exposure += curr_under_exposure

        # average measure over all evaluation sequences
        over_exposure /= len(self.eval_sequences)
        under_exposure /= len(self.eval_sequences)

        return over_exposure, under_exposure, len(self.eval_sequences), number_of_images


def main():
    with open(Path(__file__).parent.parent / "config.json", "r") as cfg_file:
        config = json.load(cfg_file)
    print(f"Calculate over- and underexposure measure for data set '{config['dataset']['name']}'.")
    print("Progress: ")

    ou_exposure = OverUnderExposure(config)
    over_exposure, under_exposure, num_sequences, num_images = ou_exposure.calculate()

    print(f"{num_sequences} sequences evaluated with {num_images} LDR images in total.")
    print(f"under-exposure measure [%]: {under_exposure}")
    print(f"over-exposure measure [%]: {over_exposure}")


if __name__ == '__main__':
    main()

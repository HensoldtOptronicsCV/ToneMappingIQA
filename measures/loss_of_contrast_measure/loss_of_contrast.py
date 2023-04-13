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

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from measures.base_measure import BaseMeasure
from measures.utils.image_preprocessing import preprocess_image

sys.path.append(str(Path(".").resolve().parent))


class LossOfContrast(BaseMeasure):
    def __init__(self, config):
        super(LossOfContrast, self).__init__(config)

        self.gamma_correction = 2.2
        
        # uncomment for CUDA support
        #self.image_ldr_gpu = cv2.cuda_GpuMat()
        #self.image_hdr_gpu = cv2.cuda_GpuMat()

    def calculate_loss_of_contrast(self, seq_name):
        """
        Calculate global and local loss of contrast measure for all HDR and LDR image pairs in given path
        :param seq_name: Name of sequence to calculate the measure for
        :return:
        """
        sequence_length = len(self.dataset_data[seq_name]["hdr"])
        hdr_images_path = self.dataset_data[seq_name]["hdr"]
        ldr_images_path = self.experiment_data[seq_name]["ldr"]

        loss_global_contrast = 0
        loss_local_contrast = 0
        with tqdm(total=sequence_length) as pbar:
            for hdr_image_path, ldr_image_path in zip(hdr_images_path, ldr_images_path):
                image_hdr = preprocess_image(cv2.imread(str(hdr_image_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH),
                                             self.norm_value_hdr)
                image_ldr = preprocess_image(cv2.imread(str(ldr_image_path), cv2.IMREAD_GRAYSCALE), self.norm_value_ldr,
                                             self.gamma_correction)

                # bring HDR and LDR image to log domain
                image_hdr = np.log10(image_hdr)
                image_ldr = np.log10(image_ldr)

                loss_global_contrast += self.measure_global_contrast(
                    image_hdr, image_ldr)
                loss_local_contrast += self.measure_local_contrast(
                    image_hdr, image_ldr)
                pbar.update(1)

        loss_global_contrast /= sequence_length
        loss_local_contrast /= sequence_length

        return loss_global_contrast, loss_local_contrast

    def measure_global_contrast(self, image_hdr, image_ldr):
        """
        Calculate the global loss of contrast measure for one given pair of frames (tone mapped image together with corresponding HDR image)
        NOTE: The images are expected to be pre-processed already (i.e. normalized and logarithmized)
        :param image_hdr: High Definition Range image (original image before tone mapping)
        :param image_ldr: Low Definition Range image (related processed image after tone mapping)
        :return:
        """
        kernel_size = 9, 9
        sigma = 3

        # calculate global contrast for the tone mapped LDR image (Eq. 4)
        image_gaussian1_global_T = cv2.GaussianBlur(
            image_ldr, kernel_size, sigma)
        image_gaussian1_global_T *= image_gaussian1_global_T
        image_gaussian2_global_T = cv2.GaussianBlur(
            image_ldr ** 2, kernel_size, sigma)
        image_gaussian_global_T = np.array(
            np.sqrt(np.abs(image_gaussian2_global_T - image_gaussian1_global_T)))
        C_global_T = -(np.sum(image_gaussian_global_T) / image_ldr.size)

        # calculate global contrast for the HDR image (Eq. 4)
        image_gaussian1_global_L = cv2.GaussianBlur(
            image_hdr, kernel_size, sigma)
        image_gaussian1_global_L *= image_gaussian1_global_L
        image_gaussian2_global_L = cv2.GaussianBlur(
            image_hdr ** 2, kernel_size, sigma)
        image_gaussian_global_L = np.array(
            np.sqrt(np.abs(image_gaussian2_global_L - image_gaussian1_global_L)))
        C_global_L = -(np.sum(image_gaussian_global_L) / image_ldr.size)

        # calculate result: loss of global contrast
        loss_global_contrast = C_global_T - C_global_L

        return loss_global_contrast

    def measure_local_contrast(self, image_hdr, image_ldr):
        """
        Calculate the local loss of contrast measure for one given pair of frames (tone mapped image together with corresponding HDR image)
        NOTE: The images are expected to be pre-processed already (i.e. normalized and logarithmized)
        :param image_hdr: High Definition Range image (original image before tone mapping)
        :param image_ldr: Low Definition Range image (related processed image after tone mapping)
        :return:
        """
        sigma_color = 0.2
        sigma_space = 10

        # --------------- ldr cuda -------------------
        # self.image_ldr_gpu.upload(image_ldr)
        # C_local1_T = cv2.cuda.bilateralFilter(
        #    self.image_ldr_gpu, -1, sigma_color, sigma_space)
        # C_local1_T = C_local1_T.download()
        # C_local2_T = cv2.cuda.subtract(image_ldr_gpu, C_local1_T)
        # C_local_T = cv2.cuda.multiply(image_ldr_gpu, cv2.cuda.abs(C_local2_T))
        # C_local_T = C_local_T.download()
        # C_local_T = np.mean(C_local_T)
        # Not worth to push the commented operations to GPU. Is slower by ~1 it/s in my experiments
        # --------------------------------------------

        # calculate local contrast for the tone mapped LDR image (Eqs. 2 and 3)
        # the parameter '-1' makes the function compute the filter size using sigma_space

        C_local1_T = cv2.bilateralFilter(image_ldr, -1, sigma_color, sigma_space)
        C_local2_T = image_ldr - C_local1_T
        C_local_T = np.mean(image_ldr * np.abs(C_local2_T))

        # --------------- hdr cuda -------------------
        # self.image_hdr_gpu.upload(image_hdr)
        # C_local1_L = cv2.cuda.bilateralFilter(
        #    self.image_hdr_gpu, -1, sigma_color, sigma_space)
        # C_local1_L = C_local1_L.download()
        # C_local2_L = cv2.cuda.subtract(image_hdr_gpu, C_local1_L)
        # C_local_L = cv2.cuda.multiply(image_hdr_gpu, cv2.cuda.abs(C_local2_L))
        # C_local_L = C_local_L.download()
        # C_local_L = np.mean(C_local_L)

        # calculate local contrast for the HDR image (Eqs. 2 and 3)
        # the parameter '-1' makes the function compute the filter size using sigma_space

        C_local1_L = cv2.bilateralFilter(image_hdr, -1, sigma_color, sigma_space)
        C_local2_L = image_hdr - C_local1_L
        C_local_L = np.mean(image_hdr * np.abs(C_local2_L))

        # calculate result: loss of local contrast
        loss_local_contrast = C_local_T - C_local_L

        return loss_local_contrast

    def calculate(self):
        global_loss_of_contrast = 0
        local_loss_of_contrast = 0
        number_of_images = 0
        for sequence_name in self.eval_sequences:
            if sequence_name not in self.dataset_data:
                raise RuntimeError(
                    f"Evaluation sequence '{sequence_name}' not found in evaluation dataset.")
            number_of_images += len(self.dataset_data[sequence_name]['hdr'])
            curr_global_loss_of_contrast, curr_local_loss_of_contrast = self.calculate_loss_of_contrast(
                sequence_name)
            global_loss_of_contrast += curr_global_loss_of_contrast
            local_loss_of_contrast += curr_local_loss_of_contrast

        # average measure over all evaluation sequences
        global_loss_of_contrast /= len(self.eval_sequences)
        local_loss_of_contrast /= len(self.eval_sequences)

        return global_loss_of_contrast, local_loss_of_contrast, len(self.eval_sequences), number_of_images


def main():
    with open(Path(__file__).parent.parent / "config.json", "r") as cfg_file:
        config = json.load(cfg_file)

    print(
        f"Calculate loss of contrast measure for data set '{config['dataset']['name']}'.")
    print("Progress: ")

    loss_of_contract = LossOfContrast(config)
    global_loss_of_contrast, local_loss_of_contrast, num_sequences, num_images = loss_of_contract.calculate()

    print(f"{num_sequences} sequences evaluated with {num_images} HDR/LDR image pairs in total.")
    print(f"Loss of global contrast: {global_loss_of_contrast}")
    print(f"Loss of local contrast: {local_loss_of_contrast}")


if __name__ == '__main__':
    main()

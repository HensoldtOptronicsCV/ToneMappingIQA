# MIT License
#
# Copyright (c) 2022 HENSOLDT
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


# Python implementation of the tone mapping quality index
# This code is related to the paper
# Hojatollah Yeganeh, Zhou Wang
# "Objective Quality Assessment of Tone-Mapped Images", 2013.


#pylint: disable-all

import json
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from scipy import stats
from scipy.signal import convolve2d, correlate2d
from tqdm import tqdm

from measures.base_measure import BaseMeasure

sys.path.append(str(Path(".").resolve().parent))


def RGBtoYxy(RGB: np.ndarray) -> np.ndarray:
    """ convert image from RGB to Yxy

    Arguments:  
        {np.ndarray} -- Input image in RGB color space as numpy array

    Return: 
        {np.ndarray} -- Output image in Yxy color space as numpy array

    """
    # Matrix for color converion: p.22 https://engineering.purdue.edu/~bouman/ece637/notes/pdf/ColorSpaces.pdf
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float32)
    # RGB image is reshaped to 2D array where each row corresponds to the rgb values of a single pixel
    RGB2 = np.reshape(RGB, (RGB.shape[0] * RGB.shape[1], 3))
    # Matrixmultiplication with conversion  matrix -> each row corresponds to the Yxy values of a single pixel
    XYZ2 = M @ RGB2.T
    S = np.sum(XYZ2, axis=0)
    Yxy2 = np.zeros_like(XYZ2)
    EPS = np.finfo(float).eps
    Yxy2[0, :] = XYZ2[1, :]
    Yxy2[1, :] = XYZ2[0, :] / (S + EPS)
    Yxy2[2, :] = XYZ2[1, :] / (S + EPS)
    Yxy = np.reshape(Yxy2.T, (RGB.shape[0], RGB.shape[1], 3))
    return Yxy


def create_gaussian_filter(ksize: int, sigma: float) -> np.ndarray:
    """ equivalent implementation of gaussian filter to matlab

    Arguments:  
        {ksize: int} -- size of filter
        {sigma: float} -- smoothing parameter

    Return: 
        {np.ndarray} --  2d gaussian filter as numpy array
    """
    kernel = np.zeros((2 * ksize + 1, 2 * ksize + 1))
    center = ksize
    for i in range(2 * ksize + 1):
        for j in range(2 * ksize + 1):
            x = i - center
            y = j - center
            kernel[i, j] = (np.exp(-0.5 * (x**2 + y**2) / sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


class TMQI(BaseMeasure):
    # definitions of the parameters are described in the paper
    def __init__(self, confiq):
        super(TMQI, self).__init__(confiq)
        self.a: float = 0.8012
        self.Alpha: float = 0.3046
        self.Beta: float = 0.7088
        self.level: int = 5
        self.weight: List[float] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.f: int = 32
        self.window: np.ndarray = create_gaussian_filter(5, 1.5)

    def Slocal(self, L_hdr: np.ndarray, L_ldr: np.ndarray, sf: float) -> float:
        """ calculate the local StructuralFidelity score between one ldr and hdr image

        Arguments:  
            {L_hdr: np.ndarray} -- Luminance channel of HDR Image  
            {L_ldr: np.ndarray} -- Luminance channel of LDR Image  
            {sf: float} -- spatial frequency
        Return: 
            {float} --  local StructuralFidelity score 
        """
        # Default parameters
        C1: float = 0.01
        C2: float = 10
        window: np.ndarray = self.window  # type: ignore

        # calculates the local mean with gaussian filter
        mu1 = convolve2d(L_hdr, window, mode='valid')
        mu2 = convolve2d(L_ldr, window, mode='valid')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        # calculates the local variance with gaussian filter
        sigma1_sq = convolve2d(L_hdr * L_hdr, window, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(L_ldr * L_ldr, window, mode='valid') - mu2_sq

        # calculates the local standard deviation
        sigma1 = np.sqrt(np.maximum(0, sigma1_sq))
        sigma2 = np.sqrt(np.maximum(0, sigma2_sq))

        # calculate cross-correlation between the HDR and LDR image patches
        sigma12 = convolve2d(L_hdr * L_ldr, window, mode='valid') - mu1_mu2

        # Mannos CSF Function
        CSF = 100 * 2.6 * (0.0192 + 0.114 * sf) * np.exp(-(0.114 * sf) ** 1.1)

        u_hdr = 128 / (1.4 * CSF)
        sig_hdr = u_hdr / 3
        sigma1p = stats.norm.cdf(sigma1, u_hdr, sig_hdr)

        u_ldr = u_hdr
        sig_ldr = u_ldr / 3

        sigma2p = stats.norm.cdf(sigma2, u_ldr, sig_ldr)

        # Definition of the local structural fidelity measure (see paper)
        s_map = (((2 * sigma1p * sigma2p) + C1) / ((sigma1p * sigma1p) +
                 (sigma2p * sigma2p) + C1)) * ((sigma12 + C2) / (sigma1 * sigma2 + C2))
        s: float = np.mean(s_map)

        return s

    def StructuralFidelity(self, L_hdr: np.ndarray, L_ldr: np.ndarray, level: int, weight: List[float]) -> float:
        """ calculate multiple StructuralFidelity scores between ldr and hdr image by downscaling image n times

        Arguments:  
            {L_hdr: np.ndarray} -- Luminance channel of HDR Image  
            {L_ldr: np.ndarray} -- Luminance channel of LDR Image  
            {level: int} -- specifies how many StructuralFidelity measurements are made
            {weight: List[float]} -- weighting list for respecitve local StructuralFidelity score

        Return: 
            {np.ndarray} -- overall StructuralFidelity score 

        """
        downsample_filter = np.ones((2, 2)) / 4
        f = self.f
        s_local_list = []
        s_local = []
        for _ in range(level):
            f = f / 2
            s_local = self.Slocal(L_hdr, L_ldr, f)
            s_local_list.append(s_local)
            filtered_im1 = correlate2d(L_hdr, downsample_filter, boundary='symm', mode='same')
            filtered_im2 = correlate2d(L_ldr, downsample_filter, boundary='symm', mode='same')
            L_hdr = filtered_im1[::2, ::2]  # downscale image by a factor of 4
            L_ldr = filtered_im2[::2, ::2]
        S = np.prod(np.power(s_local_list, weight))
        return S

    def blkproc(self, img: np.ndarray, block_size: tuple) -> np.ndarray:
        """ python implementation of matlab function blkproc. Applies a function to each block of image with sliding window

        Arguments:  
            {img: np.ndarray} -- Luminance channel of image  
            {block_size: tuple} -- size of each 2d block 

        Return: 
            {np.ndarray} -- transformed luminance channel of image  
        """
        # zero padding
        def std_function(x): return np.std(x, ddof=1) * np.ones(x.shape)
        h, w = img.shape[0:2]
        add_h = block_size[0] - (h % block_size[0])
        add_w = block_size[1] - (w % block_size[1])
        pad_image = np.pad(img, [(0, add_h), (0, add_w)], mode='constant', constant_values=0)
        rows, cols = pad_image.shape
        row_block, col_block = block_size

        # apply function operation to each block
        for row in range(0, rows, row_block):
            for col in range(0, cols, col_block):
                x = pad_image[row:row + row_block, col:col + col_block]
                pad_image[row:row + row_block, col:col + col_block] = std_function(x)
        # crop image to original size
        pad_image = pad_image[:h, :w]

        return pad_image

    def StatisticalNaturalness(self, L_ldr: np.ndarray) -> float:
        """ calculate the StatisticalNaturalness measurement for ldr image. The parameter were taken from the paper

        Arguments:  
            {L_ldr: np.ndarray} -- Luminance channel of image  

        Return: 
            {np.ndarray} -- StatisticalNaturalness measurement
        """
        u = np.mean(L_ldr)
        blocks = self.blkproc(L_ldr, (11, 11))
        blocks_mean = np.mean(blocks)
        # ------------------ Contrast measurement ----------
        par_beta = np.array([4.4, 10.1])
        beta_mode = (par_beta[0] - 1) / (par_beta[0] + par_beta[1] - 2)
        # Beta probability density function
        C_0 = stats.beta.pdf(beta_mode, par_beta[0], par_beta[1])
        C = stats.beta.pdf(blocks_mean / 64.29, par_beta[0], par_beta[1])
        pc = C / C_0
        # ----------------  Brightness measurement ---------
        mu = 115.94
        sigma = 27.99
        # Gaussian probability density functions
        B = stats.norm.pdf(u, mu, sigma)
        B_0 = stats.norm.pdf(mu, mu, sigma)
        pb = B / B_0
        N = pb * pc

        return N

    def img_preprocessing(self, hdrImage, ldrImage):
        assert hdrImage is not None is not ldrImage, "One of the images could not be loaded"
        assert hdrImage.shape[:2] == ldrImage.shape[:2], "Images have different sizes"
        assert len(hdrImage.shape) == len(ldrImage.shape), "Images are not of the same type "

        if hdrImage.ndim == 3:  # for rgb image
            HDR = RGBtoYxy(hdrImage)
            L_hdr = HDR[:, :, 0]  # extract  luminance channel
            lmin = np.min(L_hdr)
            lmax = np.max(L_hdr)

            L_hdr = np.round((2**32 - 1) / (lmax - lmin)) * (L_hdr - lmin)
            L_ldr = RGBtoYxy(ldrImage)
            L_ldr = (L_ldr[:, :, 0])

            return (L_hdr, L_ldr)

        if hdrImage.ndim == 2:  # for gray scale images or grayflag
            L_ldr = ldrImage
            lmin = np.min(hdrImage)
            lmax = np.max(hdrImage)
            # normalize luminance: range 0 to 2^32 - 1
            L_hdr = round((2**32 - 1) / (lmax - lmin)) * (hdrImage - lmin)

            return (L_hdr, L_ldr)

    def calculate_tmqi(self, seq_name, orig_implementation=True):
        sequence_length = len(self.dataset_data[seq_name]["hdr"])
        hdr_images_path = self.dataset_data[seq_name]["hdr"]
        ldr_images_path = self.experiment_data[seq_name]["ldr"]
        TMQI_val = 0
        with tqdm(total=sequence_length) as pbar:
            for hdr_image_path, ldr_image_path in zip(hdr_images_path, ldr_images_path):
                image_hdr = cv2.imread(
                    str(hdr_image_path), cv2.IMREAD_ANYDEPTH)
                image_hdr = image_hdr.astype('float64')

                image_ldr = cv2.imread(
                    str(ldr_image_path), cv2.IMREAD_ANYDEPTH)
                image_ldr = image_ldr.astype('float64')

                if orig_implementation:
                    res = self.img_preprocessing(image_hdr, image_ldr)
                    L_hdr = res[0]
                    L_ldr = res[1]
                    # %----------- structural fidelity measurement -----------------
                    S = self.StructuralFidelity(
                        L_hdr, L_ldr, self.level, self.weight)
                    # %--------- statistical naturalness measurement ---------------
                    N = self.StatisticalNaturalness(L_ldr)
                    # %------------- overall quality measurement -------------------
                    Q = self.a * (S**self.Alpha) + (1 - self.a) * (N**self.Beta)
                    print(Q)
                else:
                    raise Exception("not original implementation was removed")

                TMQI_val += Q
                pbar.update(1)

        return TMQI_val / len(hdr_images_path)

    def calculate(self):
        TMQI_val = 0
        number_of_images = 0
        for sequence_name in self.eval_sequences:
            if sequence_name not in self.dataset_data:
                raise RuntimeError(
                    f"Evaluation sequence '{sequence_name}' not found in evaluation dataset.")
            number_of_images += len(self.dataset_data[sequence_name]['hdr'])
            curr_TMQI_val = self.calculate_tmqi(sequence_name)
            TMQI_val += curr_TMQI_val

        # average measure over all evaluation sequences
        TMQI_val /= len(self.eval_sequences)

        return TMQI_val, len(self.eval_sequences), number_of_images


def main():
    with open(Path(__file__).parent.parent / "config.json", "r") as cfg_file:
        config = json.load(cfg_file)
    [tmqi_val, num_sequences, num_images] = TMQI(config).calculate()
    print(tmqi_val)


if __name__ == '__main__':
    main()

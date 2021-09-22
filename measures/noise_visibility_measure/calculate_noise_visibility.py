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
import copy
import math

# Third party imports
import cv2 as cv
import numpy as np
from scipy.interpolate import interp1d

# Local application imports
from utils import preprocess_image
from noise_visibility_measure import visual_pathway as vp
from noise_visibility_measure import steerable_pyramid as sp
from noise_visibility_measure.parameters import MetricPar, Sp3Filter

# preprocessing images and feeding them to "noise_visibility"
def calculate_noise_visibility(images_hdr_path, images_ldr_path, norm_value_hdr, norm_value_ldr):
    if len(images_hdr_path) != len(images_ldr_path):
        raise ValueError('Sizes of HDR and LDR image lists are different.')

    if len(images_hdr_path) == 0:
        raise ValueError('List of HDR or LDR image paths must not be empty.')

    noise_visibility_measure = 0
    for hdr_image_path, ldr_image_path in zip(images_hdr_path, images_ldr_path):
        print(".", end='', flush=True)  # show progress
        image_hdr = cv.imread(str(hdr_image_path), cv.IMREAD_ANYDEPTH)
        image_ldr = cv.imread(str(ldr_image_path), cv.IMREAD_ANYDEPTH)
        image_hdr = preprocess_image(image_hdr, norm_value_hdr)
        image_ldr = preprocess_image(image_ldr, norm_value_ldr)
        noise_visibility_measure += noise_visibility(image_hdr, image_ldr)

    print()  # newline after progress dots

    return noise_visibility_measure / len(images_hdr_path)


# calculating Q-metric for a given image pair
def noise_visibility(reference_image, test_image):
    metric_par = MetricPar('luminance', 30)
    metric_par.pix_per_deg = 30
    filt = Sp3Filter()

    # step 1. Visual pathway (includes optical&retinal pathway and multichannel decomposition)
    b_r, l_adapt_ref, band_freq, pad_value = vp.visual_pathway(reference_image, metric_par, filt, -1)
    b_t, l_adapt_test, _, _ = vp.visual_pathway(test_image, metric_par, filt, pad_value)

    l_adapt = (l_adapt_ref + l_adapt_test) / 2

    # precompute contrast sensitivity function (csf)
    csf_la = np.logspace(-5, 5, num=256)
    csf_log_la = np.log10(csf_la)
    b_count = b_t.sz.size

    csf = np.zeros((csf_la.size, b_count))
    for b in range(b_count):
        csf[:, b] = vp.hdrvdp_ncsf(band_freq[b], csf_la, metric_par)

    log_la = np.log10(np.clip(l_adapt, csf_la[0], csf_la[-1]))
    d_bands = copy.copy(b_t)

    # identify pixels that are different
    diff = np.divide(np.abs(np.subtract(test_image, reference_image)), np.clip(reference_image, 0.01, reference_image.max()))       # avoid division-by-zero
    diff_mask = np.zeros(diff.shape)
    for x in range(diff.shape[0]):
        diff_thrs_mask = np.where(diff > 0.001)
        diff_mask[diff_thrs_mask] = True

    Q = 0

    # for loop iterating trough spatial frequency bands of steerable pyramid object
    for b in range(b_count):
        # Masking Parameter
        p = 10 ** metric_par.mask_p
        q = 10 ** metric_par.mask_q
        pf = (10 ** metric_par.psych_func_slope) / p

        # calculating mask_xo
        mask_xo = np.zeros(sp.get_band_size(b_t, b, 0))
        for o in range(int(b_t.sz[b])):
            mask_xo += sp.mutual_masking(b, o, b_t, b_r)

        # calculating csf_b
        log_la_rs = np.clip(cv.resize(log_la, (sp.get_band_size(b_t, b, 0)[1], sp.get_band_size(b_t, b, 0)[0])),
                            csf_log_la[0], csf_log_la[-1])
        f = interp1d(csf_log_la, csf[:, b])
        csf_b = f(log_la_rs)

        band_norm = 2 ** b  # 1/nf from paper
        band_mult = 1

        for o in range(int(b_t.sz[b])):
            # get difference between test and reference image-bands
            band_diff = np.subtract(sp.get_band(b_t, b, o), sp.get_band(b_r, b, o)) * band_mult

            if b == b_count:
                n_ncsf = 1
            else:
                n_ncsf = np.divide(1, np.clip(csf_b, 0.01, csf_b.max()))            # avoid division-by-zero

            # calculating masks
            k_mask_self = 10 ** metric_par.mask_self
            k_mask_xo = 10 ** metric_par.mask_xo
            k_mask_xn = 10 ** metric_par.mask_xn
            self_mask = sp.mutual_masking(b, o, b_t, b_r)
            mask_xn = np.zeros(self_mask.shape)

            if b > 0:
                mask_xn = np.maximum(cv.resize(sp.mutual_masking(b - 1, o, b_t, b_r), (self_mask.shape[1], self_mask.shape[0])), 0) / (band_norm / 2)   # cv2.resize works with different interpolation that imresize from Matlab. Therefore final result differs from matlab original code

            if b < b_count - 2:
                mask_xn += np.maximum(
                    cv.resize(sp.mutual_masking(b + 1, o, b_t, b_r), (self_mask.shape[1], self_mask.shape[0])), 0) / (
                                       band_norm * 2)

            band_mask_xo = np.maximum(mask_xo - self_mask, 0)

            # N_Mask eq (14) from paper
            N_mask = band_norm * (k_mask_self * np.float_power((np.divide(self_mask, n_ncsf) / band_norm), q) +
                                  k_mask_xo * np.float_power(np.divide(band_mask_xo, n_ncsf) / band_norm, q) +
                                  k_mask_xn * np.float_power(np.divide(mask_xn, n_ncsf), q))

            ex_diff = np.sign(band_diff / band_norm) * np.float_power(np.abs(band_diff / band_norm), p)

            # noise-normalized signal difference eq (11) from paper
            D = ex_diff/(np.float_power(np.float_power(n_ncsf, 2*p) + np.float_power(N_mask,2), 0.5))

            d_bands = sp.set_band(d_bands, b, o, sp.sign_power(D/band_norm, pf) * band_norm)

            f = interp1d(metric_par.quality_band_freq, metric_par.quality_band_w)
            w_f = f(np.clip(band_freq[b], metric_par.quality_band_freq[-1], metric_par.quality_band_freq[0]))
            epsilon = 1e-12

            diff_mask_b = cv.resize(diff_mask, tuple([D.shape[1], D.shape[0]]))
            D_p = D * diff_mask_b

            # calculating quality measure Q for each band and adding them together
            Q += (math.log(vp.msre(D_p) + epsilon) - math.log(epsilon)) * w_f

    return 100 - Q





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


# implementation of pyrTools' steerable pyramid adjusted to this measure
# only the functions used in hdr-vdp are implemented and adjusted to this case
import numpy as np
from scipy.signal import correlate2d, convolve2d

class SteerablePyramid:
    def __init__(self, image, filt):
        self.steermtx = filt.mtx
        self.harmonics = filt.harmonics
        self.pyr, self.pind = build_Spyr(image, filt)
        self.band_freq = None
        self.sz = None


def build_Spyr(image, filt):
    hi0 = corrDn(image, filt.hi0filt, np.array([[1], [1]]), np.array([[0], [0]]))
    lo0 = corrDn(image, filt.lo0filt, np.array([[1], [1]]), np.array([[0], [0]]))
    ht = max_pyr_height(np.array(image.shape), np.array([filt.lofilt.shape[0], 1]))
    pyr, pind = build_Spyr_levels(lo0, ht, filt.lofilt, filt.bfilts)
    pyr = np.concatenate((hi0.reshape((hi0.size, 1), order='F'), pyr))
    pind = np.concatenate((np.array([[hi0.shape[0], hi0.shape[1]]]), pind))

    return pyr, pind


def build_Spyr_levels(lo0, ht, lofilt, bfilts):
    if ht <= 0:
        pyr = lo0.reshape((lo0.size, 1), order='F')
        pind = np.array([[lo0.shape[0], lo0.shape[1]]])
    else:
        bfiltsz = int(np.round(np.sqrt(bfilts.shape[0])))
        bands = np.zeros((lo0.shape[0] * lo0.shape[1], 4))
        bind = np.zeros((bfilts.shape[1], 2))
        for b in range(bfilts.shape[1]):
            filt = bfilts[:, b].reshape((bfiltsz, bfiltsz)).T
            band = corrDn(lo0, filt, np.array([[1], [1]]), np.array([[0], [0]]))
            bands[:, b] = band.reshape((band.size,), order='F')
            bind[b, :] = np.array([[band.shape]])
        lo = corrDn(lo0, lofilt, np.array([[2], [2]]), np.array([[0], [0]]))
        npyr, nind = build_Spyr_levels(lo, ht - 1, lofilt, bfilts)
        pyr = np.concatenate((bands.reshape((bands.size, 1), order='F'), npyr))
        pind = np.concatenate((bind, nind))

    return pyr, pind


def corrDn(a, b, step, start):
    if a.shape[0] >= b.shape[0] and a.shape[1] >= b.shape[1]:
        large = a
        small = b
    elif a.shape[0] <= b.shape[0] and a.shape[1] <= b.shape[1]:
        large = b
        small = a
    ly = large.shape[0]
    lx = large.shape[1]
    sy = small.shape[0]
    sx = small.shape[1]
    sy2 = int(np.floor((sy - 1) / 2))
    sx2 = int(np.floor((sx - 1) / 2))
    clarge = np.pad(large, ((sy2, sy2), (sx2, sx2)), mode='reflect')
    c = correlate2d(clarge, small, mode='valid')
    res = c[int(start[0])::int(step[0]), int(start[1])::int(step[1])]

    return res


def max_pyr_height(imsize, filtsize):  # imsize is MxN array, filtsize is scalar in this case
    if imsize[0] == 1 or imsize[1] == 1:  # 1D vector
        imsize = imsize[0] * imsize[1]
        filtsize = filtsize[0] * filtsize[-1]
    elif filtsize[0] == 1 or filtsize[1] == 1:  # 2d image, 1d filter
        filtsize = np.array([filtsize[0], filtsize[0]])
    if imsize[0] < filtsize[0] or imsize[1] < filtsize[0]:
        height = 0
    else:
        height = 1 + max_pyr_height(np.floor(imsize / 2), filtsize)

    return height


def spyr_band_num(pind):
    if pind.shape[0] == 2:
        nbands = 0
    else:
        b = 3
        compare = pind[b, :] == pind[2, :]
        while b <= pind.shape[0] and compare.all():
            compare = pind[b, :] == pind[2, :]
            b += 1
        nbands = b - 2

    return nbands


def spyr_height(pind):
    nbands = spyr_band_num(pind)
    if pind.shape[0] > 2:
        height = (pind.shape[0] - 2) / nbands
    else:
        height = 0

    return height


def pyr_band(pyr, pind, band):
    a = pyr[pyr_band_indices(pind, band)]
    s1 = int(pind[int(band) - 1, 0])
    s2 = int(pind[int(band) - 1, 1])

    return np.reshape(a, (s1, s2), order='F')


def pyr_band_indices(pind, band):
    ind = 0
    for i in range(int(band) - 1):
        ind += np.prod(pind[i, :])
    indices = np.arange(ind, ind + np.prod(pind[int(band - 1), :])).astype(int)

    return indices


def get_band_size(bands, b, o):
    temp = np.sum(bands.sz[0:b]) + o
    sz = bands.pind[np.sum(bands.sz[0:b]).astype(int) + o, :].astype(int)

    return sz


def get_band(bands, b, o):
    oc = min(o + 1, bands.sz[b])
    band = pyr_band(bands.pyr, bands.pind, int(np.sum(bands.sz[0:b]) + oc))

    return band


def mutual_masking(b, o, b_t, b_r):
    m = np.minimum(np.abs(get_band(b_t, b, o)), np.abs(get_band(b_r, b, o)))
    m = convolve2d(m, np.ones((3, 3)) / 9, mode='same')

    return m


def sign_power(x, e):
    y = np.sign(x) * np.float_power(abs(x), e)

    return y


def set_band(bands, b, o, B):
    indices = pyr_band_indices(bands.pind, sum(bands.sz[:b]) + o + 1)
    bands.pyr = bands.pyr.flatten()
    bands.pyr[np.reshape(indices, -1)] = np.reshape(B, -1, 'F')

    return bands
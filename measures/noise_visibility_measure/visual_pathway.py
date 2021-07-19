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


# Third party imports
import numpy as np
from scipy.integrate import trapz, cumtrapz
from scipy import interpolate
from scipy import fft

# Local application imports
from noise_visibility_measure import steerable_pyramid as spyr
from noise_visibility_measure.parameters import PN


# computes optical&retinal pathway and multichannel decomposition
def visual_pathway(image, metric_par, filt, pad_value):
    # ---------------------------
    # optical and retinal pathway
    # ---------------------------
    lamda, S_lmsr, rho2, mtf_filter, IMG_E = precompute_common_variables(image, metric_par)
    opt = optical_transfer(image, metric_par, mtf_filter)
    rmlsr = photoreceptor_spectral_sensitivity(opt, lamda, S_lmsr, IMG_E)
    l_adapt = adapt_luminance(rmlsr)
    plmr = photoreceptor_non_linearity(rmlsr, metric_par)
    pc, pr = remove_DC(plmr)
    p = achromatic_response(pc, pr)
    # ---------------------------
    # multichannel decomposition
    # ---------------------------
    steerPyramid = spyr.SteerablePyramid(p, filt)
    steerPyramid.sz = np.ones((int(spyr.spyr_height(steerPyramid.pind)) + 2, 1))
    steerPyramid.sz[1:-1] = spyr.spyr_band_num(steerPyramid.pind)
    pow = -np.arange(0, spyr.spyr_height(steerPyramid.pind) + 2)  # arange excludes stop
    band_freq = np.float_power(2, pow) * (metric_par.pix_per_deg / 2)
    l_mean = np.mean(l_adapt)
    bb = spyr.pyr_band(steerPyramid.pyr, steerPyramid.pind, np.sum(steerPyramid.sz))
    rho_bb = create_cycdeg_image(np.array([bb.shape[0], bb.shape[1]]) * 2, band_freq[-1] * 2 * np.sqrt(2))
    csf_bb = hdrvdp_ncsf(rho_bb, l_mean, metric_par)
    if pad_value == -1:
        pad_value = ['constant', np.mean(bb)]
    dum = fast_conv_fft(bb, csf_bb, pad_value)
    steerPyramid.pyr[spyr.pyr_band_indices(steerPyramid.pind, np.sum(steerPyramid.sz))] = np.reshape(dum, (dum.size, 1), order='F')

    return steerPyramid, l_adapt, band_freq, pad_value


# computes sigma, lamda from Eq (3) and some other parameters
def precompute_common_variables(image, metric_par):
    image_size = np.array([image.shape[0], image.shape[1]])
    rho2 = create_cycdeg_image(2 * image_size, metric_par.pix_per_deg)
    mtf_filter = hdrvdp_mtf(rho2, metric_par)

    # load spectral sensitivity curves
    lamda, S_lmsr = load_spectral_response(
        'noise_visibility_measure/data/log_cone_smith_pokorny_1975.csv')
    S_lmsr[S_lmsr == 0.0] = np.amin(S_lmsr)
    base = 10 * np.ones(S_lmsr.shape)
    S_lmsr = np.power(base, S_lmsr)
    _, S_rod = load_spectral_response('noise_visibility_measure/data/cie_scotopic_lum.txt')
    S_lmsr = np.concatenate((S_lmsr, S_rod), axis=1)
    _, IMG_E = load_spectral_response('noise_visibility_measure/data/d65.csv')
    metric_par.spectral_emission = IMG_E

    return lamda, S_lmsr, rho2, mtf_filter, IMG_E


# Eq (1) from paper
def optical_transfer(image, metric_par, mtf):
    l0 = np.zeros(image.shape)
    if metric_par.do_mtf:
        pad = ['constant', metric_par.surround_l]
        l0 = np.clip(fast_conv_fft(image[:, :], mtf, pad), 1e-5, 1e10)
    else:
        l0 = image[:, :]

    return l0


# Eq (4), also contains Eq (3)
def photoreceptor_spectral_sensitivity(l0, lamda, s_lmsr, img_e):
    m_img_lmsr = np.zeros((1, 4))  # color space transformation matrix
    lamda = lamda.reshape((1, lamda.shape[0]))
    for k in range(4):
        # this is probably for Eq (3): img_e is the emission spectra of 'f' in the equation
        temp1 = s_lmsr[:, k].reshape((s_lmsr.shape[0], 1))
        mult = np.multiply(temp1, img_e).reshape((1, temp1.shape[0]))
        m_img_lmsr[:, k] = trapz(mult, x=lamda)  # this normalized in newest version
    r_lmsr = np.reshape(np.matmul(np.reshape(l0, (l0.size, 1)), m_img_lmsr), (l0.shape[0], l0.shape[1], 4))

    return r_lmsr


# see luminance masking section: The luminance transducer functions tL|M|R(R) make an assumption that the adapting
# luminance is spatially varying and is equal to the photopic (RL + RM) or scotopic (RR) luminance of each pixel.
# This is equivalent to assuming a lack of spatial maladaptation, which is a reasonable approximation given the
# finding on spatial locus of the adaptation mechanism [He and MacLeod 1998; MacLeod et al. 1992].
#
# also in ncsf section.
def adapt_luminance(r_lmsr):
    return np.add(r_lmsr[:, :, 0], r_lmsr[:, :, 1])


# Eq (5)
# newest version includes normalization step
def photoreceptor_non_linearity(r_lmsr, metric_par):
    pn = precompute_photoreceptor_non_linearity(metric_par)
    p_lmr = np.zeros((r_lmsr.shape[0], r_lmsr.shape[1], 3))
    p_lmr[:, :, 0] = point_op(pn.Y, pn.jnd[:, 0], r_lmsr[:, :, 0])
    p_lmr[:, :, 1] = point_op(pn.Y, pn.jnd[:, 0], r_lmsr[:, :, 1])
    p_lmr[:, :, 2] = point_op(pn.Y, pn.jnd[:, 1], r_lmsr[:, :, 3])

    return p_lmr


# remove dc component
def remove_DC(plmr):
    pc = np.add(plmr[:, :, 0], plmr[:, :, 1])
    pc = np.subtract(pc, np.mean(pc))
    pr = np.subtract(plmr[:, :, 2], np.mean(plmr[:, :, 2]))

    return pc, pr


# achromatic response
def achromatic_response(pc, pr):
    return np.add(pc, pr)


# Eq (2) from paper
def hdrvdp_mtf(rho, metric_par):
    mtf = np.zeros(rho.shape)
    for kk in range(4):
        mtf = mtf + metric_par.mtf_params_a[kk] * np.exp(-metric_par.mtf_params_b[kk] * rho)

    return mtf


# used to load parameters for eq (3) from files
def load_spectral_response(filepath):
    D = np.genfromtxt(filepath, delimiter=',')
    lmin = 360
    lmax = 780
    l_step = 1
    lamda = np.linspace(lmin, lmax, int((lmax - lmin) / l_step))
    R = np.zeros((lamda.shape[0], D.shape[1] - 1))
    for k in range(1, D.shape[1]):
        f = interpolate.PchipInterpolator(D[:, 0], D[:, k], extrapolate=None)
        R[:, k - 1] = f(lamda)
        if D[0, 0] > lmin:  # because in matlab, interp1 can set extrapolated values to 0, PchipInterpolator can't
            R[:int(D[0, 0] - lmin), k - 1] = 0.0

    return lamda, R

# ---------------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------------


# convolve in fourier domain with large kernel (see hdr-vdp)
def fast_conv_fft(image, filt, pad):  # slightly different results, but satisfactory
    pad_size_0 = np.abs(np.shape(image)[0] - np.shape(filt)[0])
    pad_size_1 = np.abs(np.shape(image)[1] - np.shape(filt)[1])
    if pad[0] == 'constant':
        padded = np.pad(image, ((0, pad_size_0), (0, pad_size_1)), 'constant', constant_values=pad[1])
    else:
        padded = np.pad(image, ((0, pad_size_0), (0, pad_size_1)), pad[0])
    fx = fft.fft2(padded)
    yl = np.real(fft.ifft2(np.multiply(fx, filt)))
    y = yl[:image.shape[0], :image.shape[1]]

    return y


# low pass filter image using gaussian filter (see hdr-vdp)
def fast_gauss(image, sigma, normalize):
    if normalize:
        norm_f = 1
    else:
        norm_f = sigma * np.sqrt(2 * np.pi)
    ks = np.array([image.shape[0], image.shape[1]]) * 2
    nf = np.pi
    xx = np.concatenate([np.linspace(0, nf, int(ks[1] / 2)), np.linspace(-nf, -nf / (ks[1] / 2), int(ks[1] / 2))])
    yy = np.concatenate([np.linspace(0, nf, int(ks[0] / 2)), np.linspace(-nf, -nf / (ks[0] / 2), int(ks[0] / 2))])
    XX, YY = np.meshgrid(xx, yy)
    k = np.exp((-0.5) * np.add(np.float_power(XX, 2), np.float_power(YY, 2)) * (sigma ** 2)) * norm_f
    output = np.zeros(image.shape)
    pad = ['edge']
    for cc in range(0, image.shape[2]):
        output[:, :, cc] = fast_conv_fft(image[:, :, cc], k, pad)

    return output


# create matrix that contains frequencies, given in cycles per degree (see hdr-vdp)
def create_cycdeg_image(image_size, pix_per_degree):
    nyquist_freq = 0.5 * pix_per_degree
    half_height = np.floor(image_size[0] / 2)
    half_width = np.floor(image_size[1] / 2)
    freq_step_height = nyquist_freq / half_height
    freq_step_width = nyquist_freq / half_width
    if image_size[1] % 2 != 0:
        xx = np.concatenate([np.linspace(0, nyquist_freq, int(half_width + 1)),
                             np.linspace(-nyquist_freq, -freq_step_width, int(half_width))])
    else:
        xx = np.concatenate([np.linspace(0, nyquist_freq - freq_step_width, int(half_width)),
                             np.linspace(-nyquist_freq, -freq_step_width, int(half_width))])
    if image_size[0] % 2 != 0:
        yy = np.concatenate([np.linspace(0, nyquist_freq, int(half_height + 1)),
                             np.linspace(-nyquist_freq, -freq_step_height, int(half_height))])
    else:
        yy = np.concatenate([np.linspace(0, nyquist_freq - freq_step_height, int(half_height)),
                             np.linspace(-nyquist_freq, -freq_step_height, int(half_height))])
    XX, YY = np.meshgrid(xx, yy)
    D = np.sqrt(np.add(np.power(XX, 2), np.power(YY, 2)))

    return D


# used in photoreceptor non-linearity
def build_jnd_space_from_s(l, s):
    L = np.float_power(10, l)
    dL = np.zeros(L.shape)
    for k in range(L.shape[0]):
        thr = L[k] / s[k]
        dL[k] = (1 / thr) * L[k] * np.log(10)
    jnd = cumtrapz(dL, l, initial=0)

    return jnd


# includes Eq (6)
def precompute_photoreceptor_non_linearity(metric_par):
    cl = np.logspace(-5, 5, 2048)
    sa = hdrvdp_joint_rod_cone_sens(cl, metric_par)
    sr = hdrvdp_rod_sens(cl, metric_par) * 1
    s_max = np.subtract(sa, sr)
    s_max[s_max < 1e-3] = 1e-3
    s_min = 2 * cl
    s_min[s_min > cl[cl.shape[0] - 1]] = cl[cl.shape[0] - 1]
    f = interpolate.interp1d(cl, s_max)
    sc = 0.5 * f(s_min)
    l = np.log10(cl)
    pn = PN(l, sc, sr)
    pn.jnd = pn.jnd * metric_par.sensitivity_correction

    return pn


# used in photoreceptor non-linearity. (see hdr-vdp)
def hdrvdp_joint_rod_cone_sens(la, metric_par):
    cvi_sens_drop = metric_par.csf_sa[1]
    cvi_trans_slope = metric_par.csf_sa[2]
    cvi_low_slope = metric_par.csf_sa[3]
    s = np.power(np.add(np.float_power(np.divide(cvi_sens_drop, la), cvi_trans_slope), np.ones(la.shape)),
                 -cvi_low_slope) * metric_par.csf_sa[0]

    return s


# used in photoreceptor non linearity (see hdr-vdp)
def hdrvdp_rod_sens(la, metric_par):
    s = np.zeros(la.shape)
    peak_l = metric_par.csf_sr_par[0]
    low_s = metric_par.csf_sr_par[1]
    low_exp = metric_par.csf_sr_par[2]
    high_s = metric_par.csf_sr_par[3]
    high_exp = metric_par.csf_sr_par[4]
    rod_sens = metric_par.csf_sr_par[5]
    for i, val in enumerate(la):
        if val > peak_l:
            s[i] = np.exp(np.divide(-np.float_power(np.abs(np.log10(val / peak_l)), high_exp), high_s))
        else:
            s[i] = np.exp(np.divide(-np.float_power(np.abs(np.log10(val / peak_l)), low_exp), low_s))
    s = s * 10 ** rod_sens

    return s


# pointOp from pyrTools. Basically interpolation. Python version didn't agree with results, so had to rewrite it
def point_op(X, Y, V):
    f = interpolate.interp1d(X, Y, fill_value="extrapolate")
    temp = np.log10(np.clip(V[:, :], 10 ** X[0], 10 ** X[-1]))
    out = f(temp.reshape((temp.size, 1))).reshape((V.shape[0], V.shape[1]))

    return out

# eq (12) neural contrast sensitivity function (see hdr-vdp)
def hdrvdp_ncsf(rho, lum, metric_par):
    csf_par = metric_par.csf_params
    lum_lut = np.log10(metric_par.csf_lums)
    log_lum = np.log10(lum)
    par = np.zeros((lum.size, 4))
    for k in range(4):
        f = interpolate.interp1d(lum_lut, csf_par[:, k + 1])
        par[:, k] = f(np.clip(log_lum, lum_lut[0], lum_lut[-1]))
    a = 1+np.float_power((par[:, 0] * rho), par[:,1])
    b = np.power((1-np.exp(-np.float_power(rho / 7, 2))), par[:, 2])
    b = np.clip(b, 0.01, b.max())           # avoid division-by-zero
    s = par[:, 3]/(np.power((a/b), 0.5))

    return s


# mean square relative error
def msre(x):
    summex = sum(sum(np.float_power(x, 2)))
    numel = x.shape[0] * x.shape[1]

    return np.float_power(summex, 0.5)/numel

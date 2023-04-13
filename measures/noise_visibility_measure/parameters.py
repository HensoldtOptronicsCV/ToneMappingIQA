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

import numpy as np
from scipy.integrate import cumtrapz


class MetricPar:  # as in matlab. should also be in the website
    def __init__(self, color_encoding, pixels_per_degree):
        self.daly_peak_contrast_sens = 0.006894596
        self.sensitivity_correction = self.daly_peak_contrast_sens / 10 ** (-2.4)
        self.view_dist = 0.5
        self.spectral_emission = None
        self.orient_count = 4  # the number of orientations to consider
        # various optional features
        self.do_masking = True
        self.do_mtf = True
        self.do_spatial_pooling = True
        self.noise_model = True
        self.do_quality_raw_data = False
        self.si_size = 1.008
        # warning messages?
        self.disable_lowvals_warning = False
        self.steerpyr_filter = 'sp3Filters'
        self.mask_p = 0.544068
        self.mask_self = 0.189065
        self.mask_xo = 0.449199
        self.mask_xn = 1.52512
        self.mask_q = 0.49576
        self.si_size = -0.34244  # it is defined twice!!!
        self.psych_func_slope = np.log10(3.5)
        self.beta = self.psych_func_slope - self.mask_p
        # spatial summation
        self.si_slope = -0.850147
        self.si_sigma = -0.000502005
        self.si_ampl = 0
        # cone and rod cvi functions
        self.cvi_sens_drop = 0.0704457
        self.cvi_trans_slope = 0.0626528
        self.cvi_low_slope = - 0.00222585
        self.rod_sensitivity = 0
        self.cvi_sens_drop_rod = -0.58342
        # Achromatic csf
        self.csf_m1_f_max = 0.425509
        self.csf_m1_s_high = -0.227224
        self.csf_m1_s_low = -0.227224
        self.csf_m1_exp_low = np.log10(2)
        self.csf_params = np.array([
            [0.0160737, 0.991265, 3.74038, 0.50722, 4.46044],
            [0.383873, 0.800889, 3.54104, 0.682505, 4.94958],
            [0.929301, 0.476505, 4.37453, 0.750315, 5.28678],
            [1.29776, 0.405782, 4.40602, 0.935314, 5.61425],
            [1.49222, 0.334278, 3.79542, 1.07327, 6.4635],
            [1.46213, 0.394533, 2.7755, 1.16577, 7.45665]
        ])
        self.csf_lums = np.array([0.002, 0.02, 0.2, 2, 20, 150])
        self.csf_sa = np.array([30.162, 4.0627, 1.6596, 0.2712])
        self.csf_sr_par = np.array([1.1732, 1.1478, 1.2167, 0.5547, 2.9899, 1.1414])  # rod sensitivity function
        self.par = np.array([0.061466549455263, 0.99727370023777070])  # old parametrization of MTF
        self.mtf_params_a = np.array([self.par[1] * 0.426, self.par[1] * 0.574, (1 - self.par[1]) * self.par[0],
                                      (1 - self.par[1]) * (1 - self.par[0])])
        self.mtf_params_b = np.array([0.028, 0.37, 37, 360])
        self.quality_band_freq = np.array([60, 30, 15, 7.5, 3.75, 1.875, 0.9375, 0.4688, 0.2344, 0.1172])
        self.quality_band_w = np.array([0, 0.2832, 0.2832, 0.2142, 0.2690, 0.0398, 0.0003, 0.0003, 0, 0])
        self.quality_logistic_q1 = 3.455
        self.quality_logistic_q2 = 0.8886
        self.calibration_date = '30 Aug 2011'
        self.surround_l = 1e-5
        self.pix_per_deg = pixels_per_degree
        self.color_encoding = color_encoding


class PN:  # photoreceptor non-linearity
    def __init__(self, l, sc, sr):
        self.Y = l
        self.jnd = np.zeros((l.shape[0], 2))
        L = np.float_power(10, l)  # is it even doing something here??
        dL1 = np.zeros(L.shape)
        dL2 = np.zeros(L.shape)
        for k in range(L.shape[0]):
            thr1 = L[k] / sc[k]
            dL1[k] = (1 / thr1) * L[k] * np.log(10)
            thr2 = L[k] / sr[k]
            dL2[k] = (1 / thr2) * L[k] * np.log(10)
        self.jnd[:, 0] = cumtrapz(dL1, l, initial=0)
        self.jnd[:, 1] = cumtrapz(dL2, l, initial=0)


class Sp3Filter:
    def __init__(self):
        self.harmonics = np.array([[1, 3]])
        self.hi0filt = np.array([
            [-4.0483998600E-4, -6.2596000498E-4, -3.7829999201E-5, 8.8387000142E-4, 1.5450799838E-3, 1.9235999789E-3,
             2.0687500946E-3, 2.0898699295E-3, 2.0687500946E-3, 1.9235999789E-3, 1.5450799838E-3, 8.8387000142E-4,
             -3.7829999201E-5, -6.2596000498E-4, -4.0483998600E-4],
            [-6.2596000498E-4, -3.2734998967E-4, 7.7435001731E-4, 1.5874400269E-3, 2.1750701126E-3, 2.5626500137E-3,
             2.2892199922E-3, 1.9755100366E-3, 2.2892199922E-3, 2.5626500137E-3, 2.1750701126E-3, 1.5874400269E-3,
             7.7435001731E-4, -3.2734998967E-4, -6.2596000498E-4],
            [-3.7829999201E-5, 7.7435001731E-4, 1.1793200392E-3, 1.4050999889E-3, 2.2253401112E-3, 2.1145299543E-3,
             3.3578000148E-4, -8.3368999185E-4, 3.3578000148E-4, 2.1145299543E-3, 2.2253401112E-3, 1.4050999889E-3,
             1.1793200392E-3, 7.7435001731E-4, -3.7829999201E-5],
            [8.8387000142E-4, 1.5874400269E-3, 1.4050999889E-3, 1.2960999738E-3, -4.9274001503E-4, -3.1295299996E-3,
             -4.5751798898E-3, -5.1014497876E-3, -4.5751798898E-3, -3.1295299996E-3, -4.9274001503E-4, 1.2960999738E-3,
             1.4050999889E-3, 1.5874400269E-3, 8.8387000142E-4],
            [1.5450799838E-3, 2.1750701126E-3, 2.2253401112E-3, -4.9274001503E-4, -6.3222697936E-3, -2.7556000277E-3,
             5.3632198833E-3, 7.3032598011E-3, 5.3632198833E-3, -2.7556000277E-3, -6.3222697936E-3, -4.9274001503E-4,
             2.2253401112E-3, 2.1750701126E-3, 1.5450799838E-3],
            [1.9235999789E-3, 2.5626500137E-3, 2.1145299543E-3, -3.1295299996E-3, -2.7556000277E-3, 1.3962360099E-2,
             7.8046298586E-3, -9.3812197447E-3, 7.8046298586E-3, 1.3962360099E-2, -2.7556000277E-3, -3.1295299996E-3,
             2.1145299543E-3, 2.5626500137E-3, 1.9235999789E-3],
            [2.0687500946E-3, 2.2892199922E-3, 3.3578000148E-4, -4.5751798898E-3, 5.3632198833E-3, 7.8046298586E-3,
             -7.9501636326E-2, -0.1554141641, -7.9501636326E-2, 7.8046298586E-3, 5.3632198833E-3, -4.5751798898E-3,
             3.3578000148E-4, 2.2892199922E-3, 2.0687500946E-3],
            [2.0898699295E-3, 1.9755100366E-3, -8.3368999185E-4, -5.1014497876E-3, 7.3032598011E-3, -9.3812197447E-3,
             -0.1554141641, 0.7303866148, -0.1554141641, -9.3812197447E-3, 7.3032598011E-3, -5.1014497876E-3,
             -8.3368999185E-4, 1.9755100366E-3, 2.0898699295E-3],
            [2.0687500946E-3, 2.2892199922E-3, 3.3578000148E-4, -4.5751798898E-3, 5.3632198833E-3, 7.8046298586E-3,
             -7.9501636326E-2, -0.1554141641, -7.9501636326E-2, 7.8046298586E-3, 5.3632198833E-3, -4.5751798898E-3,
             3.3578000148E-4, 2.2892199922E-3, 2.0687500946E-3],
            [1.9235999789E-3, 2.5626500137E-3, 2.1145299543E-3, -3.1295299996E-3, -2.7556000277E-3, 1.3962360099E-2,
             7.8046298586E-3, -9.3812197447E-3, 7.8046298586E-3, 1.3962360099E-2, -2.7556000277E-3, -3.1295299996E-3,
             2.1145299543E-3, 2.5626500137E-3, 1.9235999789E-3],
            [1.5450799838E-3, 2.1750701126E-3, 2.2253401112E-3, -4.9274001503E-4, -6.3222697936E-3, -2.7556000277E-3,
             5.3632198833E-3, 7.3032598011E-3, 5.3632198833E-3, -2.7556000277E-3, -6.3222697936E-3, -4.9274001503E-4,
             2.2253401112E-3, 2.1750701126E-3, 1.5450799838E-3],
            [8.8387000142E-4, 1.5874400269E-3, 1.4050999889E-3, 1.2960999738E-3, -4.9274001503E-4, -3.1295299996E-3,
             -4.5751798898E-3, -5.1014497876E-3, -4.5751798898E-3, -3.1295299996E-3, -4.9274001503E-4, 1.2960999738E-3,
             1.4050999889E-3, 1.5874400269E-3, 8.8387000142E-4],
            [-3.7829999201E-5, 7.7435001731E-4, 1.1793200392E-3, 1.4050999889E-3, 2.2253401112E-3, 2.1145299543E-3,
             3.3578000148E-4, -8.3368999185E-4, 3.3578000148E-4, 2.1145299543E-3, 2.2253401112E-3, 1.4050999889E-3,
             1.1793200392E-3, 7.7435001731E-4, -3.7829999201E-5],
            [-6.2596000498E-4, -3.2734998967E-4, 7.7435001731E-4, 1.5874400269E-3, 2.1750701126E-3, 2.5626500137E-3,
             2.2892199922E-3, 1.9755100366E-3, 2.2892199922E-3, 2.5626500137E-3, 2.1750701126E-3, 1.5874400269E-3,
             7.7435001731E-4, -3.2734998967E-4, -6.2596000498E-4],
            [-4.0483998600E-4, -6.2596000498E-4, -3.7829999201E-5, 8.8387000142E-4, 1.5450799838E-3, 1.9235999789E-3,
             2.0687500946E-3, 2.0898699295E-3, 2.0687500946E-3, 1.9235999789E-3, 1.5450799838E-3, 8.8387000142E-4,
             -3.7829999201E-5, -6.2596000498E-4, -4.0483998600E-4]
        ])
        self.lo0filt = np.array([
            [-8.7009997515E-5, -1.3542800443E-3, -1.6012600390E-3, -5.0337001448E-4, 2.5240099058E-3, -5.0337001448E-4,
             -1.6012600390E-3, -1.3542800443E-3, -8.7009997515E-5],
            [-1.3542800443E-3, 2.9215801042E-3, 7.5227199122E-3, 8.2244202495E-3, 1.1076199589E-3, 8.2244202495E-3,
             7.5227199122E-3, 2.9215801042E-3, -1.3542800443E-3],
            [-1.6012600390E-3, 7.5227199122E-3, -7.0612900890E-3, -3.7694871426E-2, -3.2971370965E-2, -3.7694871426E-2,
             -7.0612900890E-3, 7.5227199122E-3, -1.6012600390E-3],
            [-5.0337001448E-4, 8.2244202495E-3, -3.7694871426E-2, 4.3813198805E-2, 0.1811603010, 4.3813198805E-2,
             -3.7694871426E-2, 8.2244202495E-3, -5.0337001448E-4],
            [2.5240099058E-3, 1.1076199589E-3, -3.2971370965E-2, 0.1811603010, 0.4376249909, 0.1811603010,
             -3.2971370965E-2, 1.1076199589E-3, 2.5240099058E-3],
            [-5.0337001448E-4, 8.2244202495E-3, -3.7694871426E-2, 4.3813198805E-2, 0.1811603010, 4.3813198805E-2,
             -3.7694871426E-2, 8.2244202495E-3, -5.0337001448E-4],
            [-1.6012600390E-3, 7.5227199122E-3, -7.0612900890E-3, -3.7694871426E-2, -3.2971370965E-2, -3.7694871426E-2,
             -7.0612900890E-3, 7.5227199122E-3, -1.6012600390E-3],
            [-1.3542800443E-3, 2.9215801042E-3, 7.5227199122E-3, 8.2244202495E-3, 1.1076199589E-3, 8.2244202495E-3,
             7.5227199122E-3, 2.9215801042E-3, -1.3542800443E-3],
            [-8.7009997515E-5, -1.3542800443E-3, -1.6012600390E-3, -5.0337001448E-4, 2.5240099058E-3, -5.0337001448E-4,
             -1.6012600390E-3, -1.3542800443E-3, -8.7009997515E-5]
        ])
        self.lofilt = np.array([
            [-4.3500000174E-5, 1.2078000145E-4, -6.7714002216E-4, -1.2434000382E-4, -8.0063997302E-4, -1.5970399836E-3,
             -2.5168000138E-4, -4.2019999819E-4, 1.2619999470E-3, -4.2019999819E-4, -2.5168000138E-4, -1.5970399836E-3,
             -8.0063997302E-4, -1.2434000382E-4, -6.7714002216E-4, 1.2078000145E-4, -4.3500000174E-5],
            [1.2078000145E-4, 4.4606000301E-4, -5.8146001538E-4, 5.6215998484E-4, -1.3688000035E-4, 2.3255399428E-3,
             2.8898599558E-3, 4.2872801423E-3, 5.5893999524E-3, 4.2872801423E-3, 2.8898599558E-3, 2.3255399428E-3,
             -1.3688000035E-4, 5.6215998484E-4, -5.8146001538E-4, 4.4606000301E-4, 1.2078000145E-4],
            [-6.7714002216E-4, -5.8146001538E-4, 1.4607800404E-3, 2.1605400834E-3, 3.7613599561E-3, 3.0809799209E-3,
             4.1121998802E-3, 2.2212199401E-3, 5.5381999118E-4, 2.2212199401E-3, 4.1121998802E-3, 3.0809799209E-3,
             3.7613599561E-3, 2.1605400834E-3, 1.4607800404E-3, -5.8146001538E-4, -6.7714002216E-4],
            [-1.2434000382E-4, 5.6215998484E-4, 2.1605400834E-3, 3.1757799443E-3, 3.1846798956E-3, -1.7774800071E-3,
             -7.4316998944E-3, -9.0569201857E-3, -9.6372198313E-3, -9.0569201857E-3, -7.4316998944E-3, -1.7774800071E-3,
             3.1846798956E-3, 3.1757799443E-3, 2.1605400834E-3, 5.6215998484E-4, -1.2434000382E-4],
            [-8.0063997302E-4, -1.3688000035E-4, 3.7613599561E-3, 3.1846798956E-3, -3.5306399222E-3, -1.2604200281E-2,
             -1.8847439438E-2, -1.7508180812E-2, -1.6485679895E-2, -1.7508180812E-2, -1.8847439438E-2, -1.2604200281E-2,
             -3.5306399222E-3, 3.1846798956E-3, 3.7613599561E-3, -1.3688000035E-4, -8.0063997302E-4],
            [-1.5970399836E-3, 2.3255399428E-3, 3.0809799209E-3, -1.7774800071E-3, -1.2604200281E-2, -2.0229380578E-2,
             -1.1091699824E-2, 3.9556599222E-3, 1.4385120012E-2, 3.9556599222E-3, -1.1091699824E-2, -2.0229380578E-2,
             -1.2604200281E-2, -1.7774800071E-3, 3.0809799209E-3, 2.3255399428E-3, -1.5970399836E-3],
            [-2.5168000138E-4, 2.8898599558E-3, 4.1121998802E-3, -7.4316998944E-3, -1.8847439438E-2, -1.1091699824E-2,
             2.1906599402E-2, 6.8065837026E-2, 9.0580143034E-2, 6.8065837026E-2, 2.1906599402E-2, -1.1091699824E-2,
             -1.8847439438E-2, -7.4316998944E-3, 4.1121998802E-3, 2.8898599558E-3, -2.5168000138E-4],
            [-4.2019999819E-4, 4.2872801423E-3, 2.2212199401E-3, -9.0569201857E-3, -1.7508180812E-2, 3.9556599222E-3,
             6.8065837026E-2, 0.1445499808, 0.1773651242, 0.1445499808, 6.8065837026E-2, 3.9556599222E-3,
             -1.7508180812E-2, -9.0569201857E-3, 2.2212199401E-3, 4.2872801423E-3, -4.2019999819E-4],
            [1.2619999470E-3, 5.5893999524E-3, 5.5381999118E-4, -9.6372198313E-3, -1.6485679895E-2, 1.4385120012E-2,
             9.0580143034E-2, 0.1773651242, 0.2120374441, 0.1773651242, 9.0580143034E-2, 1.4385120012E-2,
             -1.6485679895E-2, -9.6372198313E-3, 5.5381999118E-4, 5.5893999524E-3, 1.2619999470E-3],
            [-4.2019999819E-4, 4.2872801423E-3, 2.2212199401E-3, -9.0569201857E-3, -1.7508180812E-2, 3.9556599222E-3,
             6.8065837026E-2, 0.1445499808, 0.1773651242, 0.1445499808, 6.8065837026E-2, 3.9556599222E-3,
             -1.7508180812E-2, -9.0569201857E-3, 2.2212199401E-3, 4.2872801423E-3, -4.2019999819E-4],
            [-2.5168000138E-4, 2.8898599558E-3, 4.1121998802E-3, -7.4316998944E-3, -1.8847439438E-2, -1.1091699824E-2,
             2.1906599402E-2, 6.8065837026E-2, 9.0580143034E-2, 6.8065837026E-2, 2.1906599402E-2, -1.1091699824E-2,
             -1.8847439438E-2, -7.4316998944E-3, 4.1121998802E-3, 2.8898599558E-3, -2.5168000138E-4],
            [-1.5970399836E-3, 2.3255399428E-3, 3.0809799209E-3, -1.7774800071E-3, -1.2604200281E-2, -2.0229380578E-2,
             -1.1091699824E-2, 3.9556599222E-3, 1.4385120012E-2, 3.9556599222E-3, -1.1091699824E-2, -2.0229380578E-2,
             -1.2604200281E-2, -1.7774800071E-3, 3.0809799209E-3, 2.3255399428E-3, -1.5970399836E-3],
            [-8.0063997302E-4, -1.3688000035E-4, 3.7613599561E-3, 3.1846798956E-3, -3.5306399222E-3, -1.2604200281E-2,
             -1.8847439438E-2, -1.7508180812E-2, -1.6485679895E-2, -1.7508180812E-2, -1.8847439438E-2, -1.2604200281E-2,
             -3.5306399222E-3, 3.1846798956E-3, 3.7613599561E-3, -1.3688000035E-4, -8.0063997302E-4],
            [-1.2434000382E-4, 5.6215998484E-4, 2.1605400834E-3, 3.1757799443E-3, 3.1846798956E-3, -1.7774800071E-3,
             -7.4316998944E-3, -9.0569201857E-3, -9.6372198313E-3, -9.0569201857E-3, -7.4316998944E-3, -1.7774800071E-3,
             3.1846798956E-3, 3.1757799443E-3, 2.1605400834E-3, 5.6215998484E-4, -1.2434000382E-4],
            [-6.7714002216E-4, -5.8146001538E-4, 1.4607800404E-3, 2.1605400834E-3, 3.7613599561E-3, 3.0809799209E-3,
             4.1121998802E-3, 2.2212199401E-3, 5.5381999118E-4, 2.2212199401E-3, 4.1121998802E-3, 3.0809799209E-3,
             3.7613599561E-3, 2.1605400834E-3, 1.4607800404E-3, -5.8146001538E-4, -6.7714002216E-4],
            [1.2078000145E-4, 4.4606000301E-4, -5.8146001538E-4, 5.6215998484E-4, -1.3688000035E-4, 2.3255399428E-3,
             2.8898599558E-3, 4.2872801423E-3, 5.5893999524E-3, 4.2872801423E-3, 2.8898599558E-3, 2.3255399428E-3,
             -1.3688000035E-4, 5.6215998484E-4, -5.8146001538E-4, 4.4606000301E-4, 1.2078000145E-4],
            [-4.3500000174E-5, 1.2078000145E-4, -6.7714002216E-4, -1.2434000382E-4, -8.0063997302E-4, -1.5970399836E-3,
             -2.5168000138E-4, -4.2019999819E-4, 1.2619999470E-3, -4.2019999819E-4, -2.5168000138E-4, -1.5970399836E-3,
             -8.0063997302E-4, -1.2434000382E-4, -6.7714002216E-4, 1.2078000145E-4, -4.3500000174E-5]
        ])
        self.bfilts = np.array([
            [-8.1125000725E-4, 4.4451598078E-3, 1.2316980399E-2, 1.3955879956E-2, 1.4179450460E-2, 1.3955879956E-2,
             1.2316980399E-2, 4.4451598078E-3, -8.1125000725E-4,
             3.9103501476E-3, 4.4565401040E-3, -5.8724298142E-3, -2.8760801069E-3, 8.5267601535E-3, -2.8760801069E-3,
             -5.8724298142E-3, 4.4565401040E-3, 3.9103501476E-3,
             1.3462699717E-3, -3.7740699481E-3, 8.2581602037E-3, 3.9442278445E-2, 5.3605638444E-2, 3.9442278445E-2,
             8.2581602037E-3, -3.7740699481E-3, 1.3462699717E-3,
             7.4700999539E-4, -3.6522001028E-4, -2.2522680461E-2, -0.1105690673, -0.1768419296, -0.1105690673,
             -2.2522680461E-2, -3.6522001028E-4, 7.4700999539E-4,
             0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,
             0.0000000000, 0.0000000000,
             -7.4700999539E-4, 3.6522001028E-4, 2.2522680461E-2, 0.1105690673, 0.1768419296, 0.1105690673,
             2.2522680461E-2, 3.6522001028E-4, -7.4700999539E-4,
             -1.3462699717E-3, 3.7740699481E-3, -8.2581602037E-3, -3.9442278445E-2, -5.3605638444E-2, -3.9442278445E-2,
             -8.2581602037E-3, 3.7740699481E-3, -1.3462699717E-3,
             -3.9103501476E-3, -4.4565401040E-3, 5.8724298142E-3, 2.8760801069E-3, -8.5267601535E-3, 2.8760801069E-3,
             5.8724298142E-3, -4.4565401040E-3, -3.9103501476E-3,
             8.1125000725E-4, -4.4451598078E-3, -1.2316980399E-2, -1.3955879956E-2, -1.4179450460E-2, -1.3955879956E-2,
             -1.2316980399E-2, -4.4451598078E-3, 8.1125000725E-4],
            [
                0.0000000000, -8.2846998703E-4, -5.7109999034E-5, 4.0110000555E-5, 4.6670897864E-3, 8.0871898681E-3,
                1.4807609841E-2, 8.6204400286E-3, -3.1221499667E-3,
                8.2846998703E-4, 0.0000000000, -9.7479997203E-4, -6.9718998857E-3, -2.0865600090E-3, 2.3298799060E-3,
                -4.4814897701E-3, 1.4917500317E-2, 8.6204400286E-3,
                5.7109999034E-5, 9.7479997203E-4, 0.0000000000, -1.2145539746E-2, -2.4427289143E-2, 5.0797060132E-2,
                3.2785870135E-2, -4.4814897701E-3, 1.4807609841E-2,
                -4.0110000555E-5, 6.9718998857E-3, 1.2145539746E-2, 0.0000000000, -0.1510555595, -8.2495503128E-2,
                5.0797060132E-2, 2.3298799060E-3, 8.0871898681E-3,
                -4.6670897864E-3, 2.0865600090E-3, 2.4427289143E-2, 0.1510555595, 0.0000000000, -0.1510555595,
                -2.4427289143E-2, -2.0865600090E-3, 4.6670897864E-3,
                -8.0871898681E-3, -2.3298799060E-3, -5.0797060132E-2, 8.2495503128E-2, 0.1510555595, 0.0000000000,
                -1.2145539746E-2, -6.9718998857E-3, 4.0110000555E-5,
                -1.4807609841E-2, 4.4814897701E-3, -3.2785870135E-2, -5.0797060132E-2, 2.4427289143E-2, 1.2145539746E-2,
                0.0000000000, -9.7479997203E-4, -5.7109999034E-5,
                -8.6204400286E-3, -1.4917500317E-2, 4.4814897701E-3, -2.3298799060E-3, 2.0865600090E-3, 6.9718998857E-3,
                9.7479997203E-4, 0.0000000000, -8.2846998703E-4,
                3.1221499667E-3, -8.6204400286E-3, -1.4807609841E-2, -8.0871898681E-3, -4.6670897864E-3,
                -4.0110000555E-5, 5.7109999034E-5, 8.2846998703E-4, 0.0000000000],
            [
                8.1125000725E-4, -3.9103501476E-3, -1.3462699717E-3, -7.4700999539E-4, 0.0000000000, 7.4700999539E-4,
                1.3462699717E-3, 3.9103501476E-3, -8.1125000725E-4,
                -4.4451598078E-3, -4.4565401040E-3, 3.7740699481E-3, 3.6522001028E-4, 0.0000000000, -3.6522001028E-4,
                -3.7740699481E-3, 4.4565401040E-3, 4.4451598078E-3,
                -1.2316980399E-2, 5.8724298142E-3, -8.2581602037E-3, 2.2522680461E-2, 0.0000000000, -2.2522680461E-2,
                8.2581602037E-3, -5.8724298142E-3, 1.2316980399E-2,
                -1.3955879956E-2, 2.8760801069E-3, -3.9442278445E-2, 0.1105690673, 0.0000000000, -0.1105690673,
                3.9442278445E-2, -2.8760801069E-3, 1.3955879956E-2,
                -1.4179450460E-2, -8.5267601535E-3, -5.3605638444E-2, 0.1768419296, 0.0000000000, -0.1768419296,
                5.3605638444E-2, 8.5267601535E-3, 1.4179450460E-2,
                -1.3955879956E-2, 2.8760801069E-3, -3.9442278445E-2, 0.1105690673, 0.0000000000, -0.1105690673,
                3.9442278445E-2, -2.8760801069E-3, 1.3955879956E-2,
                -1.2316980399E-2, 5.8724298142E-3, -8.2581602037E-3, 2.2522680461E-2, 0.0000000000, -2.2522680461E-2,
                8.2581602037E-3, -5.8724298142E-3, 1.2316980399E-2,
                -4.4451598078E-3, -4.4565401040E-3, 3.7740699481E-3, 3.6522001028E-4, 0.0000000000, -3.6522001028E-4,
                -3.7740699481E-3, 4.4565401040E-3, 4.4451598078E-3,
                8.1125000725E-4, -3.9103501476E-3, -1.3462699717E-3, -7.4700999539E-4, 0.0000000000, 7.4700999539E-4,
                1.3462699717E-3, 3.9103501476E-3, -8.1125000725E-4],
            [
                3.1221499667E-3, -8.6204400286E-3, -1.4807609841E-2, -8.0871898681E-3, -4.6670897864E-3,
                -4.0110000555E-5, 5.7109999034E-5, 8.2846998703E-4, 0.0000000000,
                -8.6204400286E-3, -1.4917500317E-2, 4.4814897701E-3, -2.3298799060E-3, 2.0865600090E-3, 6.9718998857E-3,
                9.7479997203E-4, -0.0000000000, -8.2846998703E-4,
                -1.4807609841E-2, 4.4814897701E-3, -3.2785870135E-2, -5.0797060132E-2, 2.4427289143E-2, 1.2145539746E-2,
                0.0000000000, -9.7479997203E-4, -5.7109999034E-5,
                -8.0871898681E-3, -2.3298799060E-3, -5.0797060132E-2, 8.2495503128E-2, 0.1510555595, -0.0000000000,
                -1.2145539746E-2, -6.9718998857E-3, 4.0110000555E-5,
                -4.6670897864E-3, 2.0865600090E-3, 2.4427289143E-2, 0.1510555595, 0.0000000000, -0.1510555595,
                -2.4427289143E-2, -2.0865600090E-3, 4.6670897864E-3,
                -4.0110000555E-5, 6.9718998857E-3, 1.2145539746E-2, 0.0000000000, -0.1510555595, -8.2495503128E-2,
                5.0797060132E-2, 2.3298799060E-3, 8.0871898681E-3,
                5.7109999034E-5, 9.7479997203E-4, -0.0000000000, -1.2145539746E-2, -2.4427289143E-2, 5.0797060132E-2,
                3.2785870135E-2, -4.4814897701E-3, 1.4807609841E-2,
                8.2846998703E-4, -0.0000000000, -9.7479997203E-4, -6.9718998857E-3, -2.0865600090E-3, 2.3298799060E-3,
                -4.4814897701E-3, 1.4917500317E-2, 8.6204400286E-3,
                0.0000000000, -8.2846998703E-4, -5.7109999034E-5, 4.0110000555E-5, 4.6670897864E-3, 8.0871898681E-3,
                1.4807609841E-2, 8.6204400286E-3, -3.1221499667E-3]
        ]).T
        self.mtx = np.array([
            [0.5000, 0.3536, 0, -0.3536],
            [-0.0000, 0.3536, 0.5000, 0.3536],
            [0.5000, -0.3536, 0, 0.3536],
            [-0.0000, 0.3536, -0.5000, 0.3536]
        ])

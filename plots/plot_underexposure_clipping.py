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

import matplotlib.pyplot as plt
import numpy as np

x_FLIR = np.array([0, 0.048, 0.098, 0.148, 0.198, 0.248, 0.298, 0.348, 0.398, 0.448])
y_FLIR = np.array([0.39056396484375, 1.7227783203125, 5.8358154296875, 10.85308837890625, 15.59820556640625, 20.032958984375, 24.533935546875, 29.2755126953125, 33.99176025390625, 38.2877197265625])
e_FLIR = np.array([0.21449836364980018, 0.19506585760505596, 1.5475107581946348, 2.9482056260724443, 3.601499998621919, 3.6008095698982086, 3.4416890379971035, 3.056259046143326, 2.5032422632974853, 2.2931553506389237])

x_LTIR = np.array([0, 0.052, 0.102, 0.152, 0.202, 0.252, 0.302, 0.352, 0.402, 0.452])
y_LTIR = np.array([0.0008463541666666668, 0.011328125, 0.021875000000000006, 0.027994791666666664, 0.055789448302469145, 1.411330536265432, 9.860062210648149, 24.36434944058642, 40.074592496141975, 51.01627363040124])
e_LTIR = np.array([0.0007592385279746486, 0.021681153760476322, 0.04277418069179312, 0.0550135986779317, 0.06729710051082903, 1.7386876188572196, 13.95356284320785, 21.33151893046605, 23.578277789480406, 28.438311166911028])

plt.errorbar(x_FLIR, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x_LTIR, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('clipping')
plt.ylabel('underexposure measure [%]')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of underexposure measure')

plt.show()


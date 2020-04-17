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

x = np.array([0,0.01,0.03,0.05,0.11,0.17])

y_FLIR = np.array([8.152,16.294,19.736,21.23,23.228,24.166])
e_FLIR = np.array([0.2143,0.3702,0.2625,0.2604,0.2450,0.2439])

y_LTIR = np.array([8.5092,16.494,20.372,22.48,24.258,25.212])
e_LTIR = np.array([0.1149,1.0405,1.0474,0.2544,0.8233,0.7177])

plt.errorbar(x, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('Gaussian noise sigma')
plt.ylabel('noise visibility')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of noise visibility measure')

plt.show()

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

x = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3])

y_FLIR = np.array([0.0157,0.08251,0.2144,0.2922,0.4353,0.4366,0.5036])
e_FLIR= np.array([0.004,0.05,0.11,0.1,0.12,0.09,0.1])

y_LTIR = np.array([0.0001,0.0011,0.0025,0.071,0.1418,0.163,0.1745])
e_LTIR= np.array([0.00001,0.02,0.09,0.09,0.14,0.1,0.12])

plt.errorbar(x, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('clipping')
plt.ylabel('local temporal incoherence')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of local temporal incoherence measure')

plt.show()

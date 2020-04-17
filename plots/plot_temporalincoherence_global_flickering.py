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

x_FLIR = np.array([0,0.05,0.1,0.15,0.198,0.248,0.298])
y_FLIR = np.array([0.0000116,0.0083,0.0116,0.021,0.0264,0.036,0.039])
e_FLIR= np.array([0,0.003,0.01,0.0124,0.024,0.02,0.03])

x_LTIR = np.array([0,0.05,0.1,0.15,0.202,0.252,0.302])
y_LTIR = np.array([0.0001,0.0005,0.00096,0.0014,0.0021,0.0076,0.0093])
e_LTIR= np.array([0.00001,0.0002,0.0002,0.001,0.004,0.017,0.01])

plt.errorbar(x_FLIR, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x_LTIR, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('clipping')
plt.ylabel('global temporal incoherence')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of global temporal incoherence measure')

plt.show()

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

x = np.array([0,0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6,3.0])

y_FLIR = np.array([-0.04369083, -0.039584793, -0.03569856, -0.03279735, -0.030566689, -0.028301846, -0.026763285, -0.024763132, -0.02301808, -0.021591999])
e_FLIR = np.array([0.004421944, 0.004549285, 0.004710216, 0.0047237985, 0.004727546, 0.004808658, 0.0048366217, 0.0048488267, 0.004751098, 0.0046172077])

y_LTIR = np.array([-0.02283526, -0.021546166, -0.020002173, -0.018887043, -0.017971408, -0.016941909, -0.016235393, -0.015232864, -0.014302418, -0.013519572])
e_LTIR = np.array([0.0096077295, 0.008999671, 0.008336355, 0.007874315, 0.007502752, 0.0070949784, 0.0068151946, 0.0064178794, 0.0060492256, 0.0057424963])

plt.errorbar(x, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('Gaussian blur sigma')
plt.ylabel('local loss of contrast')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of local loss of contrast measure')

plt.show()

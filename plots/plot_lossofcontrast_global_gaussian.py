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

y_FLIR = np.array([-0.13678886539622973, -0.09716041457587449, -0.07955888648478185, -0.06878734008968183, -0.0612925539641747, -0.054131071534793684, -0.04966427859428487, -0.044404767534331625, -0.04013635081599554, -0.03685488289982958])
e_FLIR = np.array([0.03838713637214815, 0.027336907473529758, 0.022298871622201493, 0.018804348322716873, 0.016543751464415783, 0.014398938230507958, 0.013086651029821232, 0.011557688755933526, 0.010264997379997441, 0.009283502307823136])

y_LTIR = np.array([-0.03427039857979458, -0.030585994984465383, -0.026145061352826276, -0.023135326420436773, -0.020857033494413618, -0.018538257275766478, -0.017057915978016685, -0.015181225050835884, -0.013602964694584335, -0.012362687582338782])
e_LTIR = np.array([0.010664055804537007, 0.0093915158982501, 0.008165077629176584, 0.0074488068339405135, 0.006979182486796367, 0.006565860168171437, 0.006337153817063371, 0.006082997135191065, 0.005898036366851027, 0.005774648270374375])

plt.errorbar(x, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('Gaussian blur sigma')
plt.ylabel('global loss of contrast')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of global loss of contrast measure')

plt.show()

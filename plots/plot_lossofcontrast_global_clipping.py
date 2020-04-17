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

x = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45])

y_FLIR = np.array([-0.13552675930480204, -0.09129053150828968, -0.07187503765911869, -0.05806083893169552, -0.047828801263307595, -0.038599403662414475, -0.029366407234352552, -0.020752158848246823, -0.012517580633178219, -0.004518750761687759])
e_FLIR = np.array([0.035155958593696135, 0.023690789750382364, 0.018375711348724236, 0.015210248149720282, 0.012920730864491262, 0.01065503549396487, 0.00843760934799543, 0.0063427949038319395, 0.004270797144115823, 0.002217954270575022])

y_LTIR = np.array([-0.03427039857979458, -0.029349298000572427, -0.024587254739577614, -0.020117701370587583, -0.015795139219963287, -0.011615641179224008, -0.0076345008677690274, -0.0037556363662531963, 2.895137632501453e-05, 0.0037418356158559775])
e_LTIR = np.array([0.010664055804537007, 0.009049181587520171, 0.00752815611979075, 0.006074838113706453, 0.004772386107130633, 0.0036295194565140578, 0.0027123151807870237, 0.002197458147955634, 0.002256895299832764, 0.0027562822852553043])

plt.errorbar(x, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('clipping')
plt.ylabel('global loss of contrast')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of global loss of contrast measure')

plt.show()


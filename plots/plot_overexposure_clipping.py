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

x_FLIR = np.array([0.998,0.948,0.898,0.848,0.798,0.748,0.698,0.648,0.598,0.548])
y_FLIR = np.array([3.0858154296875, 7.4378662109375, 12.45904541015625, 18.06988525390625, 23.65179443359375, 29.0146484375, 34.0430908203125, 38.83251953125, 43.6820068359375, 48.197509765625])
e_FLIR = np.array([3.906727492712453, 4.792261806013946, 5.333575819384043, 5.1934277607811365, 4.2925839169208295, 3.171082593800012, 2.6869357980618633, 2.399214354709486, 2.0153543841392323, 1.9224443503356732])

x_LTIR = np.array([1.002,0.952,0.902,0.852,0.802,0.752,0.702,0.652,0.602,0.552])
y_LTIR = np.array([0.7706380208333333, 11.362890625, 13.810221354166666, 14.294531249999997, 14.605217978395064, 16.625636574074072, 23.104552469135804, 30.01083381558642, 34.708152488425924, 37.32503858024692])
e_LTIR = np.array([1.5412760416666669, 22.72561848993304, 27.617024755986087, 28.560418509133754, 28.846502410328256, 28.78929160293602, 28.150230223787595, 27.149258308059068, 26.67366804389493, 26.76606851718784])

plt.errorbar(x_FLIR, y_FLIR, e_FLIR, linestyle='solid', marker='x', color='red')
plt.errorbar(x_LTIR, y_LTIR, e_LTIR, linestyle='solid', marker='o', color='blue')

plt.xlabel('clipping')
plt.ylabel('overexposure measure [%]')
plt.legend(['FLIR', 'LTIR'])
plt.title('Verification of overexposure measure')

plt.show()

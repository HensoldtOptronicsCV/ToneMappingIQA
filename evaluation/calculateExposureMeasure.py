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

import sys
from FLIR_Thermal_Dataset_reader.FLIR_Thermal_Dataset_reader import getAllFlirSets
from exposure_measure.exposure_measure import exposure

dataReader = getAllFlirSets

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IndexError("Too few input arguments! Please provide absolute path to 'FLIR Thermal Dataset'.")

    flirThermalDatasetPath = sys.argv[1]
    train, val, video = dataReader(flirThermalDatasetPath)
    underExposure, overExposure = exposure(video['8bit'])
    print("under-exposure: ", underExposure)
    print("over-exposure: ", overExposure)

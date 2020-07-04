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
import json
from FLIR_Thermal_Dataset_reader.FLIR_Thermal_Dataset_reader import flir_thermal_dataset
from over_under_exposure_measure.over_under_exposure_measure import calculate_over_and_under_exposure_measure

def main():
    try:
        config = json.load(open("./config.json"))
    except FileNotFoundError:
        print("Could not find configuration file 'config.json'.")
        sys.exit(-1)
        
    print("Calculate over- and underexposure measure for data set '" + config['data']['name'] + "'.")
    print("Progress: ")
    
    # get evaluation data
    dataset_path = config['data']['path']
    if config['data']['name'] == "FLIR Thermal Dataset":
        data_reader = flir_thermal_dataset
    else:
        print("Currently, only 'FLIR Thermal Dataset' is supported.")
        sys.exit(-1)
    data = data_reader(dataset_path)
    
    # get evaluation sequences
    eval_sequences = config['data']['eval_sequences']
    if len(eval_sequences) == 0:
        print("No evaluation sequence provided in config file.")
        sys.exit(-1)
    
    # loop over all evaluation sequences
    over_exposure = 0
    under_exposure = 0
    number_of_images = 0
    for sequence_name in eval_sequences:
        if sequence_name not in data:
            print("Evaluation sequence '" + sequence_name + "' not found in evaluation dataset.")
            sys.exit(-1)
        number_of_images += len(data[sequence_name]['ldr'])
        curr_over_exposure, curr_under_exposure = calculate_over_and_under_exposure_measure(data[sequence_name]['ldr'])
        over_exposure += curr_over_exposure
        under_exposure += curr_under_exposure
        
    # average measure over all evaluation sequences
    over_exposure /= len(eval_sequences)
    under_exposure /= len(eval_sequences)
        
    print(str(len(eval_sequences)) + " sequences evaluated with " + str(number_of_images) + " LDR images in total.")
    print("under-exposure measure [%]: ", under_exposure)
    print("over-exposure measure [%]: ", over_exposure)


if __name__ == "__main__":
    main()

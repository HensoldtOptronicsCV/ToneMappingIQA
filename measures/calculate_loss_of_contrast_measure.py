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


import json
from FLIR_Thermal_Dataset_reader.FLIR_Thermal_Dataset_reader import flir_thermal_dataset
from loss_of_contrast_measure.loss_of_contrast_measure import calculate_loss_of_contrast

if __name__ == "__main__":
    config = json.load(open("./config.json"))
    print("Calculate loss of contrast measure for data set '{}'.".format(config['data']['name']))
    print("Progress: ")

    # get evaluation data
    dataset_path = config['data']['path']
    if config['data']['name'] == "FLIR Thermal Dataset":
        data_reader = flir_thermal_dataset
    else:
        raise RuntimeError("Currently, only 'FLIR Thermal Dataset' is supported.")
    data = data_reader(dataset_path)
    
    # get evaluation sequences
    eval_sequences = config['data']['eval_sequences']
    if len(eval_sequences) == 0:
        raise RuntimeError("No evaluation sequence provided in config file.")
        
    # get normalization values
    norm_value_hdr = config['norm_values']['hdr']
    norm_value_ldr = config['norm_values']['ldr']
    if norm_value_hdr <= 0 or norm_value_ldr <= 0:
        raise ValueError("Normalization values provided in config file are invalid.")
    
    # loop over all evaluation sequences
    global_loss_of_contrast = 0
    local_loss_of_contrast = 0
    number_of_images = 0
    for sequence_name in eval_sequences:
        if sequence_name not in data:
            raise RuntimeError("Evaluation sequence '" + sequence_name + "' not found in evaluation dataset.")
        number_of_images += len(data[sequence_name]['hdr'])
        curr_global_loss_of_contrast, curr_local_loss_of_contrast = \
            calculate_loss_of_contrast(data[sequence_name]['hdr'], data[sequence_name]['ldr'], norm_value_hdr, norm_value_ldr)
        global_loss_of_contrast += curr_global_loss_of_contrast
        local_loss_of_contrast += curr_local_loss_of_contrast
        
    # average measure over all evaluation sequences
    global_loss_of_contrast /= len(eval_sequences)
    local_loss_of_contrast /= len(eval_sequences)
    
    print("{} sequences evaluated with {} HDR/LDR image pairs in total.".format(len(eval_sequences), number_of_images))
    print("Loss of global contrast: {}".format(global_loss_of_contrast))
    print("Loss of local contrast: {}".format(local_loss_of_contrast))

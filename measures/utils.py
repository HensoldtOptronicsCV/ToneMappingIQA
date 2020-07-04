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

import numpy as np

## Preprocess a given image by (1) normalization to value range 0.0 to 1.0 and (2) gamma correction.
#  @param image Given image to be preprocessed.
#  @param norm_value Normalization value (e.g., 2^8 for 8 bit or 2^16 for 16 bit images).
#  @param gamma_correction Value for gamma correction (default value 1.0 means no gamma correction).
def preprocess_image(image, norm_value, gamma_correction : float = 1.0):
    # normalize image
    image_norm = image / np.float32(norm_value)
    # perform gamma correction
    image_norm = np.power(image_norm, gamma_correction)
    # determine the smallest value larger than zero
    hdr_minimum = np.min(np.where(image_norm > 0, image_norm, 1))
    # if the image contains values equal or smaller than zero, we replace them with hdr_minimum to avoid infinity values in log domain
    image_norm = np.where(image <= 0, hdr_minimum, image_norm)

    return np.float32(image_norm)


## Preprocess a given list of images by (1) normalization to value range 0.0 to 1.0 and (2) gamma correction.
#  @param images Given list of images to be preprocessed.
#  @param norm_value Normalization value (e.g., 2^8 for 8 bit or 2^16 for 16 bit images).
#  @param gamma_correction Value for gamma correction (default value 1.0 means no gamma correction).
def preprocess_image_list(images, norm_value, gamma_correction : float = 1.0):
    if len(images) == 0:
        raise ValueError('HDR and LDR image list must not be empty.')
    
    if norm_value <= 0:
        raise ValueError('Normalization value must be a value larger than 0.')

    # extract image dimensions to check them on-the-fly during preprocessing
    expected_dimensions = images[0].shape
    if len(expected_dimensions) > 2:
        raise ValueError('Images must be provided as grayscale and thus with only one image channel.')
        
    # preprocess images
    images_preproc = []
    for image in images:
        current_dimensions = image.shape
        if len(current_dimensions) > 2 or expected_dimensions[0] != current_dimensions[0] or expected_dimensions[1] != current_dimensions[1]:
            raise ValueError('Dimensions of images within image list do not match.')
        images_preproc.append(preprocess_image(image, norm_value, gamma_correction))
                
    return images_preproc

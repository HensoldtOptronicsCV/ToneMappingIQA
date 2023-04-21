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
import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve().parent))

import cv2
from tqdm import tqdm

from measures.base_measure import BaseMeasure
from measures.noise_visibility_measure.noise_visibility import NoiseVisibility
from measures.utils.image_preprocessing import preprocess_image


class QMetric(BaseMeasure):

    def calculate_q_metric(self, seq_name):
        hdr_images_path = self.dataset_data[seq_name]["hdr"]
        ldr_images_path = self.dataset_data[seq_name]["ldr"]
        sequence_length = len(ldr_images_path)

        noise_visibility_measure = 0
        with tqdm(total=sequence_length) as pbar:
            for hdr_image_path, ldr_image_path in zip(hdr_images_path, ldr_images_path):
                image_hdr = cv2.imread(str(hdr_image_path), cv2.IMREAD_ANYDEPTH)
                image_ldr = cv2.imread(str(ldr_image_path), cv2.IMREAD_ANYDEPTH)
                image_hdr = preprocess_image(image_hdr, self.norm_value_hdr)
                image_ldr = preprocess_image(image_ldr, self.norm_value_ldr)
                noise_visibility_measure += NoiseVisibility.noise_visibility(image_hdr, image_ldr)
                pbar.update(1)

        return noise_visibility_measure / len(hdr_images_path)

    def calculate(self):
        # loop over all evaluation sequences
        noise_visibility_measure = 0
        number_of_images = 0
        for sequence_name in self.eval_sequences:
            if sequence_name not in self.dataset_data:
                raise RuntimeError("Evaluation sequence '" + sequence_name + "' not found in evaluation dataset.")
            number_of_images += len(self.dataset_data[sequence_name]['hdr'])
            curr_noise_visibility_measure = self.calculate_q_metric(sequence_name)
            noise_visibility_measure += curr_noise_visibility_measure

        # average measure over all evaluation sequences
        return noise_visibility_measure / len(self.eval_sequences), len(self.eval_sequences), number_of_images


def main():
    with open(Path(__file__).parent.parent / "config.json", "r") as cfg_file:
        config = json.load(cfg_file)

    print("Calculate noise visibility measure for data set '{}'.".format(config['dataset']['name']))
    print("Progress: ")

    q_metric = QMetric(config)
    noise_visibility_measure, num_sequences, num_images = q_metric.calculate()

    print("{} sequences evaluated with {} HDR/LDR image pairs in total.".format(num_sequences, num_images))
    print("noise visibility: {}".format(noise_visibility_measure))


if __name__ == '__main__':
    main()

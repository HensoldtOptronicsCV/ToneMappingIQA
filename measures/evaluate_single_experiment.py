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
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from measures.loss_of_contrast_measure.loss_of_contrast import LossOfContrast
from measures.noise_visibility_measure.noise_visibility import NoiseVisibility
from measures.over_under_exposure_measure.over_under_exposure import OverUnderExposure
from measures.temporal_incoherence_measure.temporal_incoherence import TemporalIncoherence
from measures.tmqi_measure.tmqi import TMQI


def process_experiment(config):
    print(f">>>>>>>\nExperiment '{config['experiment']['name']}' Processing.")
    # computing evaluation metrics

    print(f"Calculate loss of contrast measure for data set '{config['dataset']['name']}'.")
    [global_loss_of_contrast, local_loss_of_contrast, num_sequences_1,
     num_images_1] = LossOfContrast(config).calculate()

    print(f"Calculate temporal incoherence measure for data set '{config['dataset']['name']}'.")
    [global_temporal_incoherence, local_temporal_incoherence, num_sequences_2,
     num_images_2] = TemporalIncoherence(config).calculate()

    print(f"Calculate over- and underexposure measure for data set '{config['dataset']['name']}'.")
    [over_exposure, under_exposure, num_sequences_3,
     num_images_3] = OverUnderExposure(config).calculate()

    print(f"Calculate TMQI measure for data set '{config['dataset']['name']}'.")
    [tmqi_val, num_sequences, num_images] = TMQI(config).calculate()

    print(f"Calculate noise visibility measure for data set '{config['dataset']['name']}'.")
    [noisy_visibility_measure, num_sequences_4, num_images_4] = \
        NoiseVisibility(config, load_noisy_data=True).calculate()

    assert num_images_3 == num_images_2 == num_images_1 == num_images
    assert num_sequences_3 == num_sequences_2 == num_sequences_1 == num_sequences
    assert (num_sequences_4 == num_sequences) or (np.isnan(num_sequences_4))
    assert (num_images_4 == num_images) or (np.isnan(num_images_4))
    print(f"\nExperiment '{config['experiment']['name']}' Done.")

    # write evaluation metrics to file
    date = datetime.now().strftime('%Y_%m_%d')
    filename = f"{date}_{config['experiment']['name']}_evaluation_results.txt".replace(' ', '_').lower()
    file_path = Path(".") / "results"
    if not file_path.is_dir():
        file_path.mkdir()
    with open(file_path / filename, 'w') as f:
        f.write(f"{num_sequences} sequences evaluated with {num_images} HDR/LDR image pairs in total.\n")
        f.write(f"Loss of global contrast: {global_loss_of_contrast}\n")
        f.write(f"Loss of local contrast: {local_loss_of_contrast}\n")
        f.write(f"Global temporal incoherence: {global_temporal_incoherence}\n")
        f.write(f"Local temporal incoherence: {local_temporal_incoherence}\n")
        f.write(f"under-exposure measure [%]: {under_exposure}\n")
        f.write(f"over-exposure measure [%]: {over_exposure}\n")
        f.write(f"noise visibility measure: {noisy_visibility_measure}\n")
        f.write(f"TMQI measure: {tmqi_val}\n")

    print(f"Experiment '{config['experiment']['name']}' Results written to file.\n")


def main():
    with open(Path(__file__).parent / "config.json", "r", encoding="utf-8") as cfg_file:
        config = json.load(cfg_file)
    process_experiment(config)


if __name__ == "__main__":
    main()
